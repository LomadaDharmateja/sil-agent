"""Retry with jittered exponential backoff.

``CLAUDE.md`` calls this mandatory rather than optional, and the arithmetic says
why: free tiers allow 10-15 requests per minute, and a planner-only agent makes
one call per episode with no natural pause between them. A 50-episode run will
meet HTTP 429 repeatedly. Without backoff the run does not slow down, it fails.

It lives in its own module, with the clock and the random source injected, so
the policy can be tested exhaustively in milliseconds without sleeping and
without a network. A retry policy that is only exercised against a live provider
is a retry policy nobody has actually tested.

Three decisions worth reading:

**Honour ``Retry-After`` when the server sends it.** The provider knows when its
window reopens; guessing when you have been told is how a client gets rate
limited harder. The header wins over the computed delay, subject to a cap so a
hostile or broken value cannot hang a run for an hour.

**The jitter is real, not decorative.** Plain exponential backoff makes every
blocked caller wake at the same instant and collide again — the thundering herd.
Sleeping a random duration in ``[0, delay]`` (AWS's "full jitter") spreads them
out. This matters here even single-process: the Phase 2 harness runs 40 LLM runs
back to back, and a shared key is a shared rate limit.

**Only some failures are worth retrying.** A 429 or a 503 is a moment in time; a
400 or a 401 will fail identically forever, and retrying it wastes the budget it
is meant to protect.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass


class ProviderError(Exception):
    """Base for anything that went wrong talking to a provider."""


class TransientError(ProviderError):
    """Worth retrying: rate limits, timeouts, 5xx, connection resets.

    ``retry_after`` carries the provider's own instruction when it sent one.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(ProviderError):
    """Not worth retrying: bad request, bad key, unknown model.

    Separated from TransientError so that a malformed request fails in one
    second rather than after six increasingly long sleeps. The distinction is
    the whole value of the class hierarchy.
    """


class RetriesExhausted(ProviderError):
    """Every attempt failed. Carries the last error for the record."""

    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(f"gave up after {attempts} attempts: {last}")
        self.attempts = attempts
        self.last = last


@dataclass(frozen=True)
class BackoffPolicy:
    """How hard and how long to keep trying.

    Defaults are chosen for a 15 RPM free tier: a full minute of retries, which
    covers a rate-limit window without stalling a run indefinitely.
    """

    max_attempts: int = 6
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    # A provider asking for longer than this is ignored in favour of the cap.
    # Some return Retry-After in seconds until the *daily* quota resets.
    max_retry_after_s: float = 60.0

    def delay_for(self, attempt: int, retry_after: float | None, rng: random.Random) -> float:
        """How long to wait before ``attempt`` (1-based, so the first retry is 1).

        Full jitter: a uniform draw from ``[0, ceiling]`` rather than the
        ceiling itself. The expected wait is halved *and* two callers that
        failed together no longer wake together.
        """
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.max_retry_after_s)

        ceiling = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return rng.uniform(0.0, ceiling)


# Injected so tests neither sleep nor depend on the wall clock.
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class RetryRecord:
    """What a call actually cost in attempts and waiting. Recorded, not printed."""

    attempts: int
    total_wait_s: float
    rate_limited: bool


def call_with_backoff[T](
    operation: Callable[[], T],
    *,
    policy: BackoffPolicy | None = None,
    rng: random.Random | None = None,
    sleep: Sleeper = time.sleep,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> tuple[T, RetryRecord]:
    """Call ``operation``, retrying transient failures with jittered backoff.

    Returns the result and a record of what it took. Raises
    :class:`PermanentError` immediately for failures that cannot improve, and
    :class:`RetriesExhausted` when the attempts run out.

    The caller gets a value or an exception — never a 429. Rate limiting is an
    infrastructure concern and must not reach agent code or become an episode.
    """
    active_policy = policy or BackoffPolicy()
    active_rng = rng or random.Random()

    total_wait = 0.0
    rate_limited = False
    last: BaseException | None = None

    for attempt in range(1, active_policy.max_attempts + 1):
        try:
            result = operation()
        except PermanentError:
            # Nothing about waiting will change the answer.
            raise
        except TransientError as exc:
            last = exc
            if exc.retry_after is not None:
                rate_limited = True
            if attempt == active_policy.max_attempts:
                break
            delay = active_policy.delay_for(attempt, exc.retry_after, active_rng)
            total_wait += delay
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            sleep(delay)
        else:
            return result, RetryRecord(
                attempts=attempt,
                total_wait_s=total_wait,
                rate_limited=rate_limited,
            )

    assert last is not None  # the loop only exits here after a TransientError
    raise RetriesExhausted(active_policy.max_attempts, last)


def parse_retry_after(value: str | None) -> float | None:
    """Read a ``Retry-After`` header. Returns None if absent or unparseable.

    The HTTP spec allows either a number of seconds or an HTTP date. Providers
    in practice send seconds, so only that form is handled — but a date must not
    raise, or a well-behaved server following the spec would break the client
    that asked it to wait.
    """
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except (ValueError, AttributeError):
        return None
    return seconds if seconds >= 0 else None
