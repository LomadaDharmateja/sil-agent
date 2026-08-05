"""Persistence: round-tripping and idempotency.

These tests need a real Postgres. JSONB storage and ``ON CONFLICT DO NOTHING``
on a composite key are the behaviours under test, and they are Postgres
behaviours — running them against SQLite would test SQLite.
"""

from __future__ import annotations

import pytest

from sil_agent.agent.state import (
    Best,
    Candidate,
    CandidateSource,
    CostRecord,
    Episode,
    Evaluation,
    ReplanDecision,
    RunStatus,
    SimResult,
    ToolError,
)
from sil_agent.persistence.repo import RunRepository
from sil_agent.simulators.toy import ToySimulator
from tests.conftest import make_run_state

pytestmark = pytest.mark.db


def make_episode(idx: int, objective_value: float = 1.0) -> Episode:
    return Episode(
        idx=idx,
        candidate=Candidate(
            params={"x1": 1.5, "x2": 2.5},
            rationale=f"episode {idx}",
            source=CandidateSource.BASELINE,
        ),
        result=SimResult(
            metrics={"branin": objective_value},
            objective_value=objective_value,
            feasible=True,
            wall_time_s=0.001,
        ),
        evaluation=Evaluation.computed_only(improved=True, delta_vs_best=0.0, feasible=True),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=3,
    )


# ---------------------------------------------------------------------------
# Idempotency — acceptance criterion 3
# ---------------------------------------------------------------------------


def test_appending_the_same_episode_twice_writes_one_row(repo: RunRepository) -> None:
    """The property crash recovery depends on.

    After a crash you cannot always tell whether the last write landed. The only
    safe response is to repeat it, which is only safe if repeating is harmless.
    """
    state = make_run_state()
    repo.save_run(state, simulator="branin", strategy="random_search")
    episode = make_episode(0)

    first = repo.append_episode(state.run_id, episode)
    second = repo.append_episode(state.run_id, episode)

    assert first is True, "the first insert should write a row"
    assert second is False, "the second should be a no-op, and say so"
    assert repo.count_episodes(state.run_id) == 1


def test_a_conflicting_append_does_not_overwrite_the_original(repo: RunRepository) -> None:
    """DO NOTHING, not DO UPDATE: episodes are append-only. The first write of
    an index is the permanent one."""
    state = make_run_state()
    repo.save_run(state, simulator="branin", strategy="random_search")

    repo.append_episode(state.run_id, make_episode(0, objective_value=1.0))
    repo.append_episode(state.run_id, make_episode(0, objective_value=999.0))

    episodes = repo.load_episodes(state.run_id)
    assert len(episodes) == 1
    assert episodes[0].result.objective_value == 1.0  # type: ignore[union-attr]


def test_same_index_in_different_runs_does_not_conflict(repo: RunRepository) -> None:
    """The key is (run_id, idx), not idx. Two runs both have an episode 0."""
    first_state = make_run_state()
    second_state = make_run_state()
    repo.save_run(first_state, simulator="branin", strategy="random_search")
    repo.save_run(second_state, simulator="branin", strategy="random_search")

    assert repo.append_episode(first_state.run_id, make_episode(0)) is True
    assert repo.append_episode(second_state.run_id, make_episode(0)) is True

    assert repo.count_episodes(first_state.run_id) == 1
    assert repo.count_episodes(second_state.run_id) == 1


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


def test_run_survives_a_round_trip(repo: RunRepository) -> None:
    state = make_run_state(seed=1234, episodes=33)
    repo.save_run(state, simulator="branin", strategy="random_search")

    loaded = repo.load_run(state.run_id)

    assert loaded is not None
    assert loaded.simulator == "branin"
    assert loaded.strategy == "random_search"
    assert loaded.state.run_id == state.run_id
    assert loaded.state.seed == 1234
    assert loaded.state.budget.max_evaluations == 33
    # The nested Pydantic models must survive JSONB unchanged.
    assert loaded.state.goal == state.goal


def test_constrained_goal_survives_a_round_trip(repo: RunRepository) -> None:
    """Constraints are nested models inside a JSONB column; check they come
    back as models, not as dictionaries."""
    goal = ToySimulator.from_name("branin_constrained").default_goal()
    state = make_run_state(goal=goal)
    repo.save_run(state, simulator="branin_constrained", strategy="random_search")

    loaded = repo.load_run(state.run_id)

    assert loaded is not None
    assert len(loaded.state.goal.constraints) == 1
    constraint = loaded.state.goal.constraints[0]
    assert constraint.metric == "x1_plus_x2"
    assert constraint.threshold == 10.0
    assert constraint.is_satisfied(9.0)


def test_best_survives_a_round_trip(repo: RunRepository) -> None:
    state = make_run_state()
    episode = make_episode(4, objective_value=0.5)
    assert isinstance(episode.result, SimResult)
    state_with_best = state.advanced(
        episode=episode,
        best=Best(episode_idx=4, candidate=episode.candidate, result=episode.result),
        budget=state.budget,
        status=RunStatus.EXECUTING,
    )
    repo.save_run(state_with_best, simulator="branin", strategy="random_search")

    loaded = repo.load_run(state.run_id)

    assert loaded is not None
    assert loaded.state.best is not None
    assert loaded.state.best.episode_idx == 4
    assert loaded.state.best.result.objective_value == 0.5


def test_failed_episodes_are_stored_and_come_back_as_errors(repo: RunRepository) -> None:
    """A ToolError and a SimResult share a column. The discriminator has to
    survive the round trip, or a failure would be read back as a result."""
    state = make_run_state()
    repo.save_run(state, simulator="branin", strategy="random_search")

    failed = Episode(
        idx=0,
        candidate=Candidate(params={"x1": 0.0, "x2": 0.0}, source=CandidateSource.PLANNER),
        result=ToolError(kind="GuardRejection", message="unknown parameter(s): thickness"),
        evaluation=Evaluation.computed_only(improved=False, delta_vs_best=0.0, feasible=False),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=1,
    )
    repo.append_episode(state.run_id, failed)

    episodes = repo.load_episodes(state.run_id)
    assert isinstance(episodes[0].result, ToolError)
    assert episodes[0].result.kind == "GuardRejection"
    assert episodes[0].sim_result is None


def test_episodes_come_back_in_index_order(repo: RunRepository) -> None:
    """Replaying history to recompute `best` depends on the order being right."""
    state = make_run_state()
    repo.save_run(state, simulator="branin", strategy="random_search")

    for idx in (3, 0, 4, 1, 2):  # deliberately out of order
        repo.append_episode(state.run_id, make_episode(idx))

    episodes = repo.load_episodes(state.run_id)
    assert [episode.idx for episode in episodes] == [0, 1, 2, 3, 4]


def test_loading_an_unknown_run_returns_none(repo: RunRepository) -> None:
    assert repo.load_run(make_run_state().run_id) is None


def test_save_run_updates_an_existing_row(repo: RunRepository) -> None:
    """save_run is an upsert; calling it every episode must not create duplicates."""
    state = make_run_state()
    repo.save_run(state, simulator="branin", strategy="random_search")
    repo.save_run(state.with_status(RunStatus.DONE), simulator="branin", strategy="random_search")

    loaded = repo.load_run(state.run_id)
    assert loaded is not None
    assert loaded.state.status is RunStatus.DONE


def test_load_run_includes_the_full_history(repo: RunRepository) -> None:
    state = make_run_state()
    repo.save_run(state, simulator="branin", strategy="random_search")
    for idx in range(5):
        repo.append_episode(state.run_id, make_episode(idx))

    loaded = repo.load_run(state.run_id)
    assert loaded is not None
    assert len(loaded.state.history) == 5
