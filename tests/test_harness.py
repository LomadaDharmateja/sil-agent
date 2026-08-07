"""The eval harness — deterministic run identity and restart by re-invocation.

The property under test is the one that makes a 60-run matrix survivable: an
interrupted experiment is continued by issuing the identical command again, with
no duplicated runs and no lost episodes.
"""

from __future__ import annotations

import pytest

from sil_agent.agent.state import TerminationReason
from sil_agent.eval.harness import Cell, ExperimentSpec, execute, execute_cell
from sil_agent.eval.metrics import derive_termination
from sil_agent.persistence.repo import RunRepository

pytestmark = pytest.mark.db


def small_spec(name: str = "test-experiment") -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        strategies=["random_search", "grid_search"],
        simulators=["branin"],
        seeds=[1, 2],
        max_evaluations=9,
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_run_id_is_a_function_of_the_configuration():
    first = Cell("exp", "random_search", "branin", 1, 100)
    second = Cell("exp", "random_search", "branin", 1, 100)
    assert first.run_id == second.run_id


@pytest.mark.parametrize(
    "changed",
    [
        {"experiment": "other"},
        {"strategy": "grid_search"},
        {"simulator": "hartmann6"},
        {"seed": 2},
        {"max_evaluations": 200},
    ],
)
def test_changing_any_field_changes_the_run_id(changed):
    """Including the budget.

    A 100-evaluation experiment and a 200-evaluation one must never share ids:
    they are different experiments and pooling their runs would be wrong.
    """
    base = Cell("exp", "random_search", "branin", 1, 100)
    other = Cell(**{**base.__dict__, **changed})
    assert base.run_id != other.run_id


def test_the_matrix_is_the_full_product():
    spec = small_spec()
    cells = list(spec.cells())
    assert len(cells) == spec.total_runs == 4
    assert len({cell.run_id for cell in cells}) == 4


# ---------------------------------------------------------------------------
# Execution and restart
# ---------------------------------------------------------------------------


def test_execute_runs_every_cell(repo: RunRepository):
    spec = small_spec()
    outcomes = execute(spec, repo, verbose=False)

    assert len(outcomes) == 4
    assert all(o.status == "completed" for o in outcomes)

    for cell in spec.cells():
        stored = repo.load_run(cell.run_id)
        assert stored is not None
        assert stored.experiment == spec.name
        assert derive_termination(stored) is not None


def test_re_running_the_matrix_changes_nothing(repo: RunRepository):
    """Idempotency at the level of a whole experiment.

    The second invocation must not add runs, add episodes, or alter results —
    otherwise an interrupted matrix could not simply be restarted, and a
    double-run would silently pool two sets of data into one cell.
    """
    spec = small_spec()
    execute(spec, repo, verbose=False)

    before = {
        cell.run_id: [e.candidate.params for e in repo.load_episodes(cell.run_id)]
        for cell in spec.cells()
    }

    second = execute(spec, repo, verbose=False)
    assert all(o.status == "skipped" for o in second)
    assert all(o.episodes_run == 0 for o in second)

    after = {
        cell.run_id: [e.candidate.params for e in repo.load_episodes(cell.run_id)]
        for cell in spec.cells()
    }
    assert before == after


class CrashingRepository:
    """A repository that dies just before a chosen episode is written.

    The same device as ``tests/test_resume.py``: the dangerous instant is after
    the simulation has been paid for but before anything has recorded it.
    """

    def __init__(self, inner: RunRepository, crash_before: int) -> None:
        self.inner = inner
        self.crash_before = crash_before

    def save_run(self, state, *, simulator, strategy, experiment=None):
        self.inner.save_run(
            state, simulator=simulator, strategy=strategy, experiment=experiment
        )

    def append_episode(self, run_id, episode):
        if episode.idx == self.crash_before:
            raise SimulatedCrash(f"crashing before episode {episode.idx}")
        return self.inner.append_episode(run_id, episode)

    def load_run(self, run_id):
        return self.inner.load_run(run_id)


class SimulatedCrash(Exception):
    """Stands in for the power going out mid-matrix."""


@pytest.mark.parametrize("strategy", ["random_search", "optuna_tpe"])
def test_an_interrupted_cell_is_continued_not_restarted(repo: RunRepository, strategy: str):
    """The restart case, with a real interruption.

    A cell is killed partway through, then the harness is invoked again exactly
    as before. It must continue from where it stopped, and the finished run must
    be byte-identical to one that was never interrupted.

    Parameterised over TPE deliberately: TPE is the strategy whose obvious
    implementation (a Study held on the object) breaks precisely here and
    nowhere else.
    """
    spec = ExperimentSpec(
        name="interrupted",
        strategies=[strategy],
        simulators=["branin"],
        seeds=[3],
        max_evaluations=12,
    )
    cell = next(iter(spec.cells()))

    # Reference: the same configuration, run through in one go, under a
    # different experiment name so it gets a different run_id.
    reference_spec = ExperimentSpec(
        name="reference",
        strategies=[strategy],
        simulators=["branin"],
        seeds=[3],
        max_evaluations=12,
    )
    reference_cell = next(iter(reference_spec.cells()))
    execute_cell(reference_cell, reference_spec, repo)
    reference = [e.candidate.params for e in repo.load_episodes(reference_cell.run_id)]
    assert len(reference) == 12

    # Now kill the real cell at episode 5.
    crashing = CrashingRepository(repo, crash_before=5)
    with pytest.raises(SimulatedCrash):
        execute_cell(cell, spec, crashing)  # type: ignore[arg-type]

    partial = repo.load_episodes(cell.run_id)
    assert len(partial) == 5, "episodes 0..4 were durably committed before the crash"

    stored = repo.load_run(cell.run_id)
    assert stored is not None
    assert derive_termination(stored) is None, "an unfinished run must not look finished"

    # Re-invoke exactly as before. No resume flag, no extra argument.
    outcome = execute_cell(cell, spec, repo)
    assert outcome.status == "completed"
    assert outcome.episodes_run == 7, "continued at 5, not restarted at 0"

    finished = [e.candidate.params for e in repo.load_episodes(cell.run_id)]
    assert finished == reference


def test_grid_search_exhaustion_is_recorded(repo: RunRepository):
    """A grid that finishes early leaves an EXHAUSTED run, not a BUDGET one."""
    spec = ExperimentSpec(
        name="grid-exhaustion",
        strategies=["grid_search"],
        simulators=["hartmann6"],
        seeds=[1],
        max_evaluations=100,
    )
    outcomes = execute(spec, repo, verbose=False)

    assert outcomes[0].result is not None
    assert outcomes[0].result.reason is TerminationReason.EXHAUSTED

    stored = repo.load_run(outcomes[0].cell.run_id)
    assert stored is not None
    assert derive_termination(stored) is TerminationReason.EXHAUSTED
    # 100 ** (1/6) is 2.15, so two points per axis: 64 evaluations of 100.
    assert len(stored.state.history) == 64


# ---------------------------------------------------------------------------
# Run locking (Phase 4)
# ---------------------------------------------------------------------------
#
# Phase 3 built `services/locks.py` for exactly the collision Phase 3.5 then
# hit — a killed shell script whose child `ablate` process survived, followed
# by a re-issued command, so two processes walked the same experiment. Nothing
# was corrupted, because the `(run_id, idx)` natural key refused the duplicate
# insert, but the loser had already paid for a simulation and a model call.
#
# These tests cover the wiring rather than Redis: `build_lock` is substituted,
# so the suite still runs with nothing installed. The lock's own semantics —
# SET NX EX, the compare-and-delete, the stale holder — are covered against a
# fake Redis in `test_stagnation_and_locks.py`.


class RecordingLock:
    """A lock that records what the harness did with it."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.acquired = False
        self.released = False
        self.renewals = 0

    def acquire(self) -> None:
        if not self.available:
            from sil_agent.services.locks import LockUnavailable

            raise LockUnavailable("held by another worker")
        self.acquired = True

    def renew(self) -> bool:
        self.renewals += 1
        return True

    def release(self) -> None:
        self.released = True


def test_a_cell_held_by_another_worker_is_skipped_not_failed(repo: RunRepository, monkeypatch):
    """The whole matrix must not stop because one cell is busy.

    `RunLock.acquire` never blocks, so moving on is the cheap option — and the
    right one, since a matrix is a list of independent cells.
    """
    lock = RecordingLock(available=False)
    monkeypatch.setattr("sil_agent.eval.harness.build_lock", lambda run_id: lock)

    spec = small_spec("locked-experiment")
    cell = next(spec.cells())

    outcome = execute_cell(cell, spec, repo)

    assert outcome.status == "locked"
    assert outcome.episodes_run == 0
    assert repo.count_episodes(cell.run_id) == 0, "nothing was written"


def test_the_lock_is_held_for_the_run_and_released_after(repo: RunRepository, monkeypatch):
    lock = RecordingLock()
    monkeypatch.setattr("sil_agent.eval.harness.build_lock", lambda run_id: lock)

    spec = small_spec("locked-experiment")
    cell = next(spec.cells())

    outcome = execute_cell(cell, spec, repo)

    assert outcome.status == "completed"
    assert lock.acquired and lock.released
    # One renewal per durably written episode: the lease is extended as work is
    # committed, rather than a TTL being chosen long enough for the whole run.
    assert lock.renewals == spec.max_evaluations


def test_the_lock_is_released_even_when_the_loop_raises(repo: RunRepository, monkeypatch):
    """A crashed worker must not leave the run locked until the TTL expires."""
    lock = RecordingLock()
    monkeypatch.setattr("sil_agent.eval.harness.build_lock", lambda run_id: lock)

    def explode(**kwargs):
        raise RuntimeError("loop exploded")

    monkeypatch.setattr("sil_agent.eval.harness.run_loop", explode)

    spec = small_spec("locked-experiment")
    with pytest.raises(RuntimeError, match="loop exploded"):
        execute_cell(next(spec.cells()), spec, repo)

    assert lock.released
