"""Run locking, so two workers cannot write to the same run.

Deferred from Phase 1, which noted the gap honestly: the loop already *detects*
a collision, because the second writer's episode insert conflicts on
``(run_id, idx)`` and returns False, and the run stops with a clear message. Safe
but not graceful — the loser has already paid for a simulation, and from Phase 3
for a model call too.

The lock makes the collision impossible rather than merely survivable.

**How it works.** ``SET key value NX EX ttl`` — set only if absent, with an
expiry — is atomic in Redis, so exactly one caller can win regardless of timing.
The value is a token unique to this holder.

**Why the token matters.** Releasing must be conditional on still owning the
lock. Without that check, a worker that stalled past the TTL would delete a lock
a *different* worker had since acquired, and both would then be writing. The
release is a small Lua script because "compare then delete" has to be one atomic
operation — doing it as two commands reintroduces the race it exists to close.

**Why the TTL needs a real number now.** In Phase 1 an episode was 13 ms, so any
TTL was generous. From Phase 3 an episode is a model call plus retries and
backoff, which can legitimately take a minute. A TTL shorter than the longest
plausible episode would expire under a healthy worker and let a second one in.
Hence the heartbeat: long work extends the lease rather than needing a TTL long
enough to cover the worst case, which would leave a crashed run locked for that
long.
"""

from __future__ import annotations

import os
import socket
import uuid
from types import TracebackType
from typing import Protocol
from uuid import UUID

# Compare-and-delete, atomically. KEYS[1] is the lock key, ARGV[1] the token.
# Returns 1 if this holder owned the lock and released it, 0 otherwise.
RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# Compare-and-extend. Renewing unconditionally would let a worker that had
# already lost the lock push out the new owner's expiry.
RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
else
    return 0
end
"""


class LockUnavailable(Exception):
    """Another worker holds this run."""


class RedisLike(Protocol):
    """The slice of the Redis client this needs.

    A protocol rather than the concrete client so tests can substitute a fake
    and the whole suite keeps running with no Redis.

    Both methods return ``object`` rather than a narrower type. That is not
    laziness: ``redis.Redis.set`` really is declared as
    ``bool | str | bytes | None`` (it returns the previous value when called
    with ``get=True``), and ``eval`` returns whatever the Lua script produced.
    Declaring anything narrower here made the real client fail to satisfy this
    protocol — mypy caught it. Both call sites below only ever ask whether the
    result is truthy, so ``object`` is exactly as much as is actually known.
    """

    def set(
        self, name: str, value: str, *, nx: bool = ..., ex: int | None = ...
    ) -> object: ...

    def eval(self, script: str, numkeys: int, *args: str) -> object: ...


def lock_key(run_id: UUID) -> str:
    return f"sil_agent:run_lock:{run_id}"


def make_token() -> str:
    """Identify this holder well enough to debug a stuck lock.

    Host and process ID make a held lock traceable to a real process; the random
    suffix keeps two runs in the same process distinct.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class RunLock:
    """An exclusive lease on one run, with a renewable TTL.

    Usable as a context manager:

        with RunLock(client, run_id):
            run_loop(...)
    """

    def __init__(
        self,
        client: RedisLike,
        run_id: UUID,
        *,
        ttl_s: int = 120,
        token: str | None = None,
    ) -> None:
        """``ttl_s`` must exceed the longest plausible single episode.

        120 seconds covers a model call plus the full retry budget. It is not
        the whole run — the heartbeat extends the lease — so a crashed worker
        frees the run in two minutes rather than in however long the run would
        have taken.
        """
        if ttl_s < 1:
            raise ValueError("ttl_s must be at least 1 second")
        self._client = client
        self._run_id = run_id
        self._ttl_s = ttl_s
        self._token = token or make_token()
        self._held = False

    @property
    def token(self) -> str:
        return self._token

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        """Take the lock, or raise LockUnavailable. Never blocks.

        Failing fast rather than waiting: the caller is a batch run, and a run
        already being worked on by someone else should be skipped, not queued
        behind.
        """
        acquired = self._client.set(lock_key(self._run_id), self._token, nx=True, ex=self._ttl_s)
        if not acquired:
            raise LockUnavailable(
                f"run {self._run_id} is locked by another worker. "
                "If that worker is gone, the lock expires on its own."
            )
        self._held = True

    def renew(self) -> bool:
        """Extend the lease. Returns False if the lock was lost.

        A False here is important rather than cosmetic: it means this worker no
        longer owns the run and must stop writing to it immediately.
        """
        if not self._held:
            return False
        extended = self._client.eval(
            RENEW_SCRIPT, 1, lock_key(self._run_id), self._token, str(self._ttl_s)
        )
        if not extended:
            self._held = False
        return bool(extended)

    def release(self) -> None:
        """Give up the lock, but only if it is still ours."""
        if not self._held:
            return
        self._client.eval(RELEASE_SCRIPT, 1, lock_key(self._run_id), self._token)
        self._held = False

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class NullLock:
    """A lock that always succeeds. The default when Redis is not configured.

    Phase 1 and 2 ran fine without locking, and requiring Redis to run a single
    local baseline would be a regression. The collision detection in
    ``append_episode`` is still there underneath, so the unlocked path remains
    safe — just less graceful, which is exactly what Phase 1 documented.
    """

    def __init__(self, run_id: UUID) -> None:
        self._run_id = run_id
        self.token = "null-lock"
        self.held = True

    def acquire(self) -> None:
        return None

    def renew(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def __enter__(self) -> NullLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def build_lock(run_id: UUID, *, ttl_s: int = 120) -> RunLock | NullLock:
    """A real lock when ``REDIS_URL`` is set and reachable, otherwise a null one.

    Degrading rather than failing is deliberate. Locking protects against a
    second *worker*; a developer running one command on a laptop has no second
    worker, and should not need Redis running to execute a benchmark.
    """
    url = os.environ.get("REDIS_URL")
    if not url:
        return NullLock(run_id)

    try:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
    except Exception:
        # Redis configured but unreachable. The alternative — failing the run —
        # would make an optional safety feature a hard dependency.
        return NullLock(run_id)

    return RunLock(client, run_id, ttl_s=ttl_s)
