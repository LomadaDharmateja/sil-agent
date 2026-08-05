"""The Phase 2 changes to the loop's exits.

Phase 1 terminated on `step_idx >= max_evaluations` — episodes run, not
evaluations spent. Those two agree exactly as long as every episode reaches the
simulator, which is true of a sampler that cannot propose anything invalid, so
the difference was invisible. It stops being invisible in Phase 3.

These tests use a strategy that deliberately proposes nonsense, which nothing in
Phases 1-2 otherwise does, to pin the behaviour down before an LLM depends on it.
"""

from __future__ import annotations

import random
from uuid import UUID

from sil_agent.agent.loop import run_loop
from sil_agent.agent.state import (
    BudgetState,
    Candidate,
    CandidateSource,
    Episode,
    Goal,
    RunState,
    RunStatus,
    TerminationReason,
    utcnow,
)
from sil_agent.simulators.toy import ToySimulator
from sil_agent.strategies.base import StrategyExhausted
from sil_agent.strategies.grid import GridSearch
from sil_agent.strategies.random_search import RandomSearch


class InMemoryRepository:
    """Satisfies the loop's repository protocol with a dictionary.

    The loop declares only the two methods it needs, so this is accepted without
    inheriting from anything and without a database being anywhere near the test.
    """

    def __init__(self) -> None:
        self.episodes: dict[tuple[UUID, int], Episode] = {}
        self.saves = 0

    def save_run(self, state: RunState, *, simulator: str, strategy: str) -> None:
        self.saves += 1

    def append_episode(self, run_id: UUID, episode: Episode) -> bool:
        key = (run_id, episode.idx)
        if key in self.episodes:
            return False
        self.episodes[key] = episode
        return True


class AlwaysInventsAParameter:
    """A proposer that hallucinates. Stands in for a Phase 3 planner having a bad day."""

    @property
    def name(self) -> str:
        return "bad_proposer"

    def propose(self, goal: Goal, history: list[Episode], rng: random.Random) -> Candidate:
        return Candidate(params={"not_a_real_parameter": 1.0}, source=CandidateSource.PLANNER)


class ExhaustsImmediately:
    @property
    def name(self) -> str:
        return "exhausted"

    def propose(self, goal: Goal, history: list[Episode], rng: random.Random) -> Candidate:
        raise StrategyExhausted("nothing to propose")


def make_state(*, max_evaluations: int, max_rejections: int = 50) -> RunState:
    simulator = ToySimulator.from_name("branin")
    now = utcnow()
    return RunState(
        run_id=UUID("11111111-2222-3333-4444-555555555555"),
        goal=simulator.default_goal(),
        status=RunStatus.PENDING,
        history=[],
        best=None,
        budget=BudgetState(max_evaluations=max_evaluations, max_rejections=max_rejections),
        step_idx=0,
        seed=1,
        created_at=now,
        updated_at=now,
    )


def test_rejections_do_not_consume_the_evaluation_budget():
    """The fairness rule the whole ablation rests on.

    A strategy that proposes ten pieces of nonsense and then ten valid points
    must still get its full ten simulator calls. If rejections came out of the
    same allowance, a hallucinating planner would lose the comparison partly
    because of the accounting rather than because of the search.
    """
    repo = InMemoryRepository()
    result = run_loop(
        state=make_state(max_evaluations=5, max_rejections=3),
        simulator=ToySimulator.from_name("branin"),
        strategy=AlwaysInventsAParameter(),
        repo=repo,
    )

    assert result.reason is TerminationReason.REJECTIONS
    assert result.state.budget.evaluations_used == 0, "no simulator call was ever made"
    assert result.state.budget.rejections_used == 3
    assert len(repo.episodes) == 3, "every rejection is still recorded in the history"


def test_a_valid_strategy_spends_exactly_its_evaluation_budget():
    repo = InMemoryRepository()
    result = run_loop(
        state=make_state(max_evaluations=7),
        simulator=ToySimulator.from_name("branin"),
        strategy=RandomSearch(),
        repo=repo,
    )

    assert result.reason is TerminationReason.BUDGET
    assert result.state.budget.evaluations_used == 7
    assert len(repo.episodes) == 7


def test_strategy_exhaustion_terminates_cleanly():
    repo = InMemoryRepository()
    result = run_loop(
        state=make_state(max_evaluations=10),
        simulator=ToySimulator.from_name("branin"),
        strategy=ExhaustsImmediately(),
        repo=repo,
    )

    assert result.reason is TerminationReason.EXHAUSTED
    assert result.state.status is RunStatus.DONE, "running out of grid is not a failure"
    assert len(repo.episodes) == 0


def test_grid_search_exhausts_before_its_budget_in_six_dimensions():
    """The curse of dimensionality, end to end through the loop.

    A 200-evaluation budget buys 2 points per axis in 6-D, so grid search covers
    all 64 and stops with 136 evaluations unspent. The report has to be able to
    say that, rather than showing grid search losing a contest it never finished
    playing.
    """
    simulator = ToySimulator.from_name("hartmann6")
    now = utcnow()
    state = RunState(
        run_id=UUID("99999999-2222-3333-4444-555555555555"),
        goal=simulator.default_goal(),
        status=RunStatus.PENDING,
        history=[],
        best=None,
        budget=BudgetState(max_evaluations=200),
        step_idx=0,
        seed=1,
        created_at=now,
        updated_at=now,
    )

    result = run_loop(
        state=state,
        simulator=simulator,
        strategy=GridSearch(200),
        repo=InMemoryRepository(),
    )

    assert result.reason is TerminationReason.EXHAUSTED
    assert result.episodes_run == 64
    assert result.state.budget.remaining_evaluations == 136
