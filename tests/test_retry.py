"""Retry and backoff.

Every test here runs in microseconds and touches no network: the clock and the
random source are injected, so "wait 30 seconds" is an entry in a list rather
than a delay. A retry policy exercised only against a live provider is a policy
nobody has tested — the interesting cases (six consecutive 429s, a hostile
`Retry-After`, a permanent failure) are exactly the ones that are hard to
provoke on demand.
"""

from __future__ import annotations

import random

import pytest

from sil_agent.services.retry import (
    BackoffPolicy,
    PermanentError,
    RetriesExhausted,
    TransientError,
    call_with_backoff,
    parse_retry_after,
)


class Recorder:
    """Stands in for `time.sleep`, remembering what it was asked to wait."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def failing(times: int, error: Exception, then: str = "ok"):
    """An operation that raises `times` times, then succeeds."""
    state = {"calls": 0}

    def operation() -> str:
        state["calls"] += 1
        if state["calls"] <= times:
            raise error
        return then

    operation.state = state  # type: ignore[attr-defined]
    return operation


# ---------------------------------------------------------------------------
# The happy path and the ordinary retry
# ---------------------------------------------------------------------------


def test_success_on_the_first_attempt_never_sleeps():
    sleeper = Recorder()
    result, record = call_with_backoff(lambda: "ok", sleep=sleeper)

    assert result == "ok"
    assert record.attempts == 1
    assert sleeper.waits == []


def test_retries_a_transient_failure_and_succeeds():
    sleeper = Recorder()
    operation = failing(2, TransientError("429 slow down"))

    result, record = call_with_backoff(
        operation, sleep=sleeper, rng=random.Random(1), policy=BackoffPolicy()
    )

    assert result == "ok"
    assert record.attempts == 3
    assert len(sleeper.waits) == 2, "one sleep per retry, none after success"


def test_a_429_never_reaches_the_caller():
    """The point of the module.

    Rate limiting is an infrastructure concern. It must not surface as an
    exception in agent code, and it must not become a recorded episode.
    """
    operation = failing(5, TransientError("429", retry_after=0.5))
    result, record = call_with_backoff(operation, sleep=Recorder(), rng=random.Random(0))

    assert result == "ok"
    assert record.rate_limited is True


# ---------------------------------------------------------------------------
# Backoff shape
# ---------------------------------------------------------------------------


def test_delays_grow_exponentially_within_their_ceiling():
    """Full jitter draws from [0, ceiling], so assert the ceiling, not the value."""
    policy = BackoffPolicy(max_attempts=6, base_delay_s=1.0, max_delay_s=30.0)
    rng = random.Random(7)

    for attempt, ceiling in [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 16.0)]:
        for _ in range(50):
            assert 0.0 <= policy.delay_for(attempt, None, rng) <= ceiling


def test_delay_is_capped():
    policy = BackoffPolicy(base_delay_s=1.0, max_delay_s=30.0)
    rng = random.Random(3)
    # 2 ** 19 would be 524,288 seconds without the cap.
    assert all(policy.delay_for(20, None, rng) <= 30.0 for _ in range(100))


def test_jitter_actually_varies():
    """Without this the thundering herd is back.

    Plain exponential backoff makes every blocked caller wake at the same
    instant and collide again. This asserts the delays are genuinely spread,
    not that a `random` call exists somewhere.
    """
    policy = BackoffPolicy()
    rng = random.Random(11)
    delays = {policy.delay_for(4, None, rng) for _ in range(50)}

    assert len(delays) > 40, "delays should be near-unique, not a constant"


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------


def test_retry_after_overrides_the_computed_delay():
    """The provider knows when its window reopens; guessing is worse."""
    policy = BackoffPolicy()
    assert policy.delay_for(1, retry_after=12.0, rng=random.Random(0)) == 12.0


def test_a_hostile_retry_after_is_capped():
    """Some providers return seconds until the *daily* quota resets.

    Obeying that literally would hang a run for hours.
    """
    policy = BackoffPolicy(max_retry_after_s=60.0)
    assert policy.delay_for(1, retry_after=86_400.0, rng=random.Random(0)) == 60.0


@pytest.mark.parametrize(
    ("header", "expected"),
    [("5", 5.0), ("0", 0.0), ("2.5", 2.5), (None, None), ("", None), ("-1", None)],
)
def test_parse_retry_after(header, expected):
    assert parse_retry_after(header) == expected


def test_an_http_date_retry_after_does_not_raise():
    """The spec allows a date. Providers send seconds, but a client must not
    break when a server follows the spec it was pointed at."""
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


# ---------------------------------------------------------------------------
# Giving up, and not trying at all
# ---------------------------------------------------------------------------


def test_permanent_failures_are_not_retried():
    """A 401 will fail identically forever. Six sleeps would waste a minute."""
    sleeper = Recorder()
    operation = failing(1, PermanentError("401 invalid api key"))

    with pytest.raises(PermanentError):
        call_with_backoff(operation, sleep=sleeper)

    assert operation.state["calls"] == 1
    assert sleeper.waits == []


def test_retries_are_exhausted_and_carry_the_last_error():
    sleeper = Recorder()
    operation = failing(99, TransientError("503 upstream down"))

    with pytest.raises(RetriesExhausted) as caught:
        call_with_backoff(
            operation, policy=BackoffPolicy(max_attempts=4), sleep=sleeper, rng=random.Random(0)
        )

    assert caught.value.attempts == 4
    assert isinstance(caught.value.last, TransientError)
    assert operation.state["calls"] == 4
    assert len(sleeper.waits) == 3, "no sleep after the final failed attempt"


def test_the_retry_record_totals_the_waiting():
    operation = failing(3, TransientError("429", retry_after=2.0))
    _, record = call_with_backoff(operation, sleep=Recorder(), rng=random.Random(0))

    assert record.attempts == 4
    assert record.total_wait_s == pytest.approx(6.0)  # three retries at 2.0 each


def test_on_retry_callback_sees_every_attempt():
    seen: list[int] = []
    operation = failing(2, TransientError("429", retry_after=1.0))

    call_with_backoff(
        operation,
        sleep=Recorder(),
        rng=random.Random(0),
        on_retry=lambda attempt, delay, exc: seen.append(attempt),
    )

    assert seen == [1, 2]
