"""Durable execution — the point of the whole phase.

Two levels of proof:

* **In-process** — a repository that raises instead of writing, simulating a
  crash at the worst possible instant. Fast, and runs in CI.
* **Subprocess** — the real CLI, killed with ``os._exit`` (no cleanup, no
  flush, as close to ``kill -9`` as a process can do to itself) and then
  resumed. Slower, but it exercises the actual entry point rather than a
  test harness.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from uuid import UUID

import pytest

from sil_agent.agent.loop import (
    RunRepositoryProtocol,
    rehydrate,
    run_loop,
)
from sil_agent.agent.state import Episode, RunState, RunStatus
from sil_agent.persistence.repo import RunRepository
from sil_agent.simulators.toy import ToySimulator
from sil_agent.strategies.random_search import RandomSearch
from tests.conftest import make_run_state

pytestmark = pytest.mark.db


class SimulatedCrash(Exception):
    """Stands in for the power going out."""


class CrashingRepository:
    """Wraps a real repository and dies just before a chosen episode is written.

    That instant is the dangerous one: the simulation has already been run and
    paid for, but nothing has recorded it. If the design were wrong, this is
    where work would be lost or duplicated.

    Note this satisfies ``RunRepositoryProtocol`` without inheriting from
    anything — the loop accepts it because it has the right methods.
    """

    def __init__(self, inner: RunRepository, crash_before: int) -> None:
        self.inner = inner
        self.crash_before = crash_before

    def save_run(self, state: RunState, *, simulator: str, strategy: str) -> None:
        self.inner.save_run(state, simulator=simulator, strategy=strategy)

    def append_episode(self, run_id: UUID, episode: Episode) -> bool:
        if episode.idx == self.crash_before:
            raise SimulatedCrash(f"crashing before episode {episode.idx}")
        return self.inner.append_episode(run_id, episode)


def objective_sequence(episodes: list[Episode]) -> list[float]:
    values = []
    for episode in episodes:
        result = episode.sim_result
        assert result is not None
        values.append(result.objective_value)
    return values


def param_sequence(episodes: list[Episode]) -> list[dict[str, float | int | str]]:
    return [episode.candidate.params for episode in episodes]


def run_fresh(
    repo: RunRepositoryProtocol,
    *,
    sim: str = "branin",
    episodes: int = 20,
    seed: int = 42,
    state: RunState | None = None,
) -> RunState:
    simulator = ToySimulator.from_name(sim)
    run_state = state or make_run_state(goal=simulator.default_goal(), episodes=episodes, seed=seed)
    repo.save_run(run_state, simulator=sim, strategy="random_search")
    result = run_loop(
        state=run_state,
        simulator=simulator,
        strategy=RandomSearch(),
        repo=repo,
    )
    return result.state


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — a run persists exactly the episodes it ran
# ---------------------------------------------------------------------------


def test_a_twenty_episode_run_persists_twenty_rows(repo: RunRepository) -> None:
    state = run_fresh(repo, episodes=20)
    assert repo.count_episodes(state.run_id) == 20
    assert [e.idx for e in repo.load_episodes(state.run_id)] == list(range(20))


def test_a_completed_run_is_marked_done(repo: RunRepository) -> None:
    state = run_fresh(repo, episodes=5)
    loaded = repo.load_run(state.run_id)
    assert loaded is not None
    assert loaded.state.status is RunStatus.DONE


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — crash and resume
# ---------------------------------------------------------------------------


def test_crash_at_ten_leaves_exactly_ten_episodes(repo: RunRepository) -> None:
    state = make_run_state(episodes=20)
    repo.save_run(state, simulator="branin", strategy="random_search")

    with pytest.raises(SimulatedCrash):
        run_loop(
            state=state,
            simulator=ToySimulator.from_name("branin"),
            strategy=RandomSearch(),
            repo=CrashingRepository(repo, crash_before=10),
        )

    # Episodes 0-9 are durable; episode 10 was never recorded, so it will be
    # re-run rather than skipped. Not 11 rows, and not 0.
    assert repo.count_episodes(state.run_id) == 10


def test_resume_continues_from_the_first_unwritten_episode(repo: RunRepository) -> None:
    state = make_run_state(episodes=20)
    repo.save_run(state, simulator="branin", strategy="random_search")

    with pytest.raises(SimulatedCrash):
        run_loop(
            state=state,
            simulator=ToySimulator.from_name("branin"),
            strategy=RandomSearch(),
            repo=CrashingRepository(repo, crash_before=10),
        )

    stored = repo.load_run(state.run_id)
    assert stored is not None
    resumed = rehydrate(stored.state)
    assert resumed.step_idx == 10, "resume must start at 10 — not 0, and not 11"

    run_loop(
        state=resumed,
        simulator=ToySimulator.from_name("branin"),
        strategy=RandomSearch(),
        repo=repo,
    )

    episodes = repo.load_episodes(state.run_id)
    assert len(episodes) == 20
    assert [e.idx for e in episodes] == list(range(20)), "no gaps, no duplicates"


def test_a_resumed_run_is_identical_to_one_that_never_crashed(repo: RunRepository) -> None:
    """The strongest statement of the property: the interruption leaves no trace.

    This is what per-episode RNG seeding buys. With a single generator carried
    across episodes, the resumed half of the run would draw a different sequence
    and this assertion would fail.
    """
    uninterrupted_state = make_run_state(episodes=20, seed=7)
    run_fresh(repo, episodes=20, seed=7, state=uninterrupted_state)
    expected = repo.load_episodes(uninterrupted_state.run_id)

    crashed_state = make_run_state(episodes=20, seed=7)
    repo.save_run(crashed_state, simulator="branin", strategy="random_search")
    with pytest.raises(SimulatedCrash):
        run_loop(
            state=crashed_state,
            simulator=ToySimulator.from_name("branin"),
            strategy=RandomSearch(),
            repo=CrashingRepository(repo, crash_before=10),
        )
    stored = repo.load_run(crashed_state.run_id)
    assert stored is not None
    run_loop(
        state=rehydrate(stored.state),
        simulator=ToySimulator.from_name("branin"),
        strategy=RandomSearch(),
        repo=repo,
    )
    actual = repo.load_episodes(crashed_state.run_id)

    assert param_sequence(actual) == param_sequence(expected)
    assert objective_sequence(actual) == objective_sequence(expected)


def test_resuming_a_finished_run_does_nothing(repo: RunRepository) -> None:
    """Idempotent at the run level too: resume twice, still 5 episodes."""
    state = run_fresh(repo, episodes=5)

    stored = repo.load_run(state.run_id)
    assert stored is not None
    result = run_loop(
        state=rehydrate(stored.state),
        simulator=ToySimulator.from_name("branin"),
        strategy=RandomSearch(),
        repo=repo,
    )

    assert result.episodes_run == 0
    assert repo.count_episodes(state.run_id) == 5


def test_best_is_recomputed_not_trusted(repo: RunRepository) -> None:
    """Rule 1: the episodes are the truth, the run row is a convenience.

    A deliberately corrupted snapshot must not survive a reload.
    """
    state = run_fresh(repo, episodes=10)
    stored = repo.load_run(state.run_id)
    assert stored is not None
    real_best = stored.state.best
    assert real_best is not None

    # Corrupt the snapshot: claim no best was ever found, and that we are at 0.
    corrupted = RunState(
        run_id=stored.state.run_id,
        goal=stored.state.goal,
        status=stored.state.status,
        history=stored.state.history,
        best=None,
        budget=stored.state.budget,
        step_idx=0,
        seed=stored.state.seed,
        created_at=stored.state.created_at,
        updated_at=stored.state.updated_at,
    )

    repaired = rehydrate(corrupted)

    assert repaired.step_idx == 10
    assert repaired.best is not None
    assert repaired.best.episode_idx == real_best.episode_idx
    assert repaired.best.result.objective_value == real_best.result.objective_value


# ---------------------------------------------------------------------------
# Acceptance criterion 5 — determinism
# ---------------------------------------------------------------------------


def test_two_runs_with_the_same_seed_are_identical(repo: RunRepository) -> None:
    first = run_fresh(repo, episodes=15, seed=42)
    second = run_fresh(repo, episodes=15, seed=42)

    assert param_sequence(repo.load_episodes(first.run_id)) == param_sequence(
        repo.load_episodes(second.run_id)
    )


def test_different_seeds_give_different_runs(repo: RunRepository) -> None:
    """Otherwise the previous test would pass for the wrong reason."""
    first = run_fresh(repo, episodes=15, seed=42)
    second = run_fresh(repo, episodes=15, seed=43)

    assert param_sequence(repo.load_episodes(first.run_id)) != param_sequence(
        repo.load_episodes(second.run_id)
    )


# ---------------------------------------------------------------------------
# The same thing, through the real CLI, with a real process kill
# ---------------------------------------------------------------------------


def _cli(*args: str, database_url: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # Point the CLI at the test database rather than the development one.
    env["DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "-m", "sil_agent.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_run_can_be_killed_and_resumed(repo: RunRepository, test_database_url: str) -> None:
    """Acceptance criterion 2, end to end, with a real process death.

    --crash-at calls os._exit(1) after the simulation has run but before the
    episode is written. No exception handling, no cleanup, no flush.
    """
    crashed = _cli(
        "run",
        "--sim",
        "branin",
        "--episodes",
        "20",
        "--seed",
        "42",
        "--crash-at",
        "10",
        database_url=test_database_url,
    )
    assert crashed.returncode == 1, crashed.stderr

    match = re.search(r"run_id: ([0-9a-f-]{36})", crashed.stdout)
    assert match is not None, f"could not find run_id in output:\n{crashed.stdout}"
    run_id = UUID(match.group(1))

    assert repo.count_episodes(run_id) == 10

    resumed = _cli("resume", "--run-id", str(run_id), database_url=test_database_url)
    assert resumed.returncode == 0, resumed.stderr
    assert "resuming at episode 10 of 20" in resumed.stdout

    episodes = repo.load_episodes(run_id)
    assert [e.idx for e in episodes] == list(range(20))

    # And the interrupted run matches an uninterrupted one with the same seed.
    clean = _cli(
        "run",
        "--sim",
        "branin",
        "--episodes",
        "20",
        "--seed",
        "42",
        database_url=test_database_url,
    )
    assert clean.returncode == 0, clean.stderr
    clean_match = re.search(r"run_id: ([0-9a-f-]{36})", clean.stdout)
    assert clean_match is not None
    clean_episodes = repo.load_episodes(UUID(clean_match.group(1)))

    assert param_sequence(episodes) == param_sequence(clean_episodes)


def test_cli_show_reports_a_run(repo: RunRepository, test_database_url: str) -> None:
    started = _cli(
        "run",
        "--sim",
        "branin_constrained",
        "--episodes",
        "6",
        "--seed",
        "3",
        database_url=test_database_url,
    )
    assert started.returncode == 0, started.stderr
    match = re.search(r"run_id: ([0-9a-f-]{36})", started.stdout)
    assert match is not None
    run_id = match.group(1)

    shown = _cli("show", "--run-id", run_id, "--episodes", database_url=test_database_url)
    assert shown.returncode == 0, shown.stderr
    assert "episodes:   6 of 6" in shown.stdout
    assert "constraint: x1_plus_x2 LE 10.0" in shown.stdout
    assert "step_idx:   stored=6  recomputed=6" in shown.stdout
