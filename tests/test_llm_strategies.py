"""The LLM strategies, driven through the real loop with a fake model.

No network and no API key: the router is replaced by an object that returns
whatever the test scripts. That is the point — the whole suite has to run
offline, and the interesting cases (a hallucinated parameter, a model that will
not produce JSON, a batch shorter than the budget) are ones a real provider
cannot be relied upon to produce on demand.

The loop, the guard, the budget accounting and the persistence are all the real
ones. Only the model is fake.
"""

from __future__ import annotations

import math
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from sil_agent.agent.loop import run_loop
from sil_agent.agent.planner import BatchProposal, PlannerProposal, describe_history
from sil_agent.agent.state import (
    BudgetState,
    Candidate,
    CandidateSource,
    CostRecord,
    Episode,
    Evaluation,
    ReplanDecision,
    RunState,
    RunStatus,
    TerminationReason,
    ToolError,
    utcnow,
)
from sil_agent.services.replay import CachingRouter, call_key
from sil_agent.services.router import LLMOutputError, Prompt, Role
from sil_agent.simulators.toy import ToySimulator
from sil_agent.strategies.llm_agent import AgentNoReflection, SingleShotLLM
from tests.test_loop_termination import InMemoryRepository


class ScriptedRouter:
    """A ModelRouter that returns pre-built objects instead of calling a model."""

    def __init__(self, replies: list[BaseModel | Exception]) -> None:
        self._replies = list(replies)
        self.prompts: list[Prompt] = []
        self.calls = 0

    def complete(self, role: Role, prompt: Prompt, schema: type[BaseModel]):
        self.calls += 1
        self.prompts.append(prompt)
        reply = self._replies.pop(0) if self._replies else self._replies_default(schema)
        if isinstance(reply, Exception):
            raise reply
        return reply, CostRecord(calls=1, prompt_tokens=100, completion_tokens=20, model="fake:m")

    @staticmethod
    def _replies_default(schema: type[BaseModel]) -> BaseModel:
        if schema is BatchProposal:
            return BatchProposal(proposals=[PlannerProposal(params={"x1": 1.0, "x2": 1.0})])
        return PlannerProposal(params={"x1": 1.0, "x2": 1.0}, rationale="default")


def proposal(x1: float, x2: float, why: str = "because") -> PlannerProposal:
    return PlannerProposal(params={"x1": x1, "x2": x2}, rationale=why)


def make_state(*, max_evaluations: int, max_rejections: int = 50) -> RunState:
    simulator = ToySimulator.from_name("branin")
    now = utcnow()
    return RunState(
        run_id=uuid4(),
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


def drive(strategy, state: RunState, repo=None):
    return run_loop(
        state=state,
        simulator=ToySimulator.from_name("branin"),
        strategy=strategy,
        repo=repo or InMemoryRepository(),
    )


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_the_agent_runs_end_to_end_through_the_real_loop():
    router = ScriptedRouter([proposal(1.0, 1.0), proposal(math.pi, 2.275), proposal(0.0, 5.0)])
    agent = AgentNoReflection(router, max_evaluations=3)

    result = drive(agent, make_state(max_evaluations=3))

    assert result.reason is TerminationReason.BUDGET
    assert result.episodes_run == 3
    assert router.calls == 3, "one model call per episode"
    assert result.state.best is not None
    # (pi, 2.275) is one of Branin's three global minima. The agent found it
    # because the test handed it to the model — this asserts the plumbing
    # carries a proposal intact from the model to the simulator to `best`, not
    # that the model is clever.
    assert result.state.best.result.objective_value == pytest.approx(0.397887, abs=1e-5)


def test_llm_cost_is_recorded_on_every_episode():
    """Cost reaches the episode through the optional ReportsCost protocol.

    Baselines do not implement it and record zero; this asserts the LLM path
    actually populates it, which is what Phase 6's budget governor will enforce
    against.
    """
    router = ScriptedRouter([proposal(1.0, 1.0), proposal(2.0, 2.0)])
    repo = InMemoryRepository()

    drive(AgentNoReflection(router, max_evaluations=2), make_state(max_evaluations=2), repo)

    costs = [episode.cost for episode in repo.episodes.values()]
    assert all(c.calls == 1 for c in costs)
    assert all(c.prompt_tokens == 100 for c in costs)
    assert all(c.model == "fake:m" for c in costs)


# ---------------------------------------------------------------------------
# The guard, now load-bearing
# ---------------------------------------------------------------------------


def test_a_hallucinated_parameter_is_rejected_and_costs_no_evaluation():
    """The fairness rule, with a real hallucination.

    The guard was written in Phase 1 against a sampler that could not produce a
    bad candidate. This is the first time anything actually exercises it.
    """
    router = ScriptedRouter(
        [
            PlannerProposal(params={"x1": 1.0, "velocity": 9.0}),  # invented
            proposal(2.0, 2.0),
            proposal(3.0, 3.0),
        ]
    )
    repo = InMemoryRepository()

    result = drive(
        AgentNoReflection(router, max_evaluations=2), make_state(max_evaluations=2), repo
    )

    assert result.state.budget.evaluations_used == 2, "the rejection did not eat an evaluation"
    assert result.state.budget.rejections_used == 1
    assert len(repo.episodes) == 3, "the rejection is still recorded in the history"

    failed = repo.episodes[(result.state.run_id, 0)]
    assert failed.sim_result is None
    assert "velocity" in failed.result.message


def test_out_of_bounds_values_are_clamped_not_rejected():
    """A near-miss is worth evaluating; the clamp is recorded, not silent."""
    router = ScriptedRouter([PlannerProposal(params={"x1": 999.0, "x2": -50.0})])
    repo = InMemoryRepository()

    result = drive(
        AgentNoReflection(router, max_evaluations=1), make_state(max_evaluations=1), repo
    )

    stored = repo.episodes[(result.state.run_id, 0)]
    assert stored.candidate.params == {"x1": 10.0, "x2": 0.0}  # clamped to the declared bounds
    assert stored.sim_result is not None, "a clamped candidate still reaches the simulator"


def test_repeated_hallucination_terminates_the_run():
    """A permanently confused planner must not loop forever.

    It ends on the rejection allowance, which the report shows distinctly from
    running out of budget.
    """
    router = ScriptedRouter([PlannerProposal(params={"nonsense": 1.0})] * 10)
    agent = AgentNoReflection(router, max_evaluations=50)

    result = drive(agent, make_state(max_evaluations=50, max_rejections=3))

    assert result.reason is TerminationReason.REJECTIONS
    assert result.state.budget.evaluations_used == 0


# ---------------------------------------------------------------------------
# A model that will not produce usable output
# ---------------------------------------------------------------------------


def test_an_unusable_model_reply_becomes_an_episode_not_a_crash():
    """The router has already retried and repaired. This is what is left."""
    router = ScriptedRouter(
        [LLMOutputError("params: field required"), proposal(1.0, 1.0), proposal(2.0, 2.0)]
    )
    repo = InMemoryRepository()

    result = drive(
        AgentNoReflection(router, max_evaluations=2), make_state(max_evaluations=2), repo
    )

    assert result.reason is TerminationReason.BUDGET
    assert result.state.budget.rejections_used == 1
    failed = repo.episodes[(result.state.run_id, 0)]
    assert failed.candidate.params == {}, "no proposal was made, and the record says so"
    assert "field required" in failed.result.message


# ---------------------------------------------------------------------------
# What the planner is shown
# ---------------------------------------------------------------------------


def test_the_planner_sees_history_and_the_incumbent():
    router = ScriptedRouter([proposal(1.0, 1.0), proposal(2.0, 2.0), proposal(3.0, 3.0)])
    drive(AgentNoReflection(router, max_evaluations=3), make_state(max_evaluations=3))

    first, third = router.prompts[0].user, router.prompts[2].user
    assert "Nothing has been evaluated yet" in first
    assert "Best so far" in third
    assert "x1=1" in third, "earlier proposals must appear in the history block"


def test_rejections_are_shown_back_to_the_planner():
    """A planner told nothing about its mistake repeats it.

    This is the one piece of feedback a *planner-only* agent gets, and it is not
    reflection — it is the guard's verdict, computed deterministically.
    """
    goal = ToySimulator.from_name("branin").default_goal()
    rejected = Episode(
        idx=0,
        candidate=Candidate(params={"velocity": 1.0}, source=CandidateSource.PLANNER),
        result=ToolError(kind="GuardRejection", message="unknown parameter(s): velocity"),
        evaluation=Evaluation.computed_only(
            improved=False, delta_vs_best=0.0, feasible=False
        ),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=1,
    )

    text = describe_history([rejected], goal)
    assert "REJECTED" in text
    assert "velocity" in text


# ---------------------------------------------------------------------------
# Single shot, and why it is stateless
# ---------------------------------------------------------------------------


def test_single_shot_asks_once_and_serves_the_batch(monkeypatch):
    """With a caching router the batch is regenerated but paid for once.

    The strategy holds nothing between calls — that is Rule 1 — and the cache
    turns the re-derivation into a database read.
    """
    batch = BatchProposal(proposals=[proposal(1.0, 1.0), proposal(2.0, 2.0), proposal(3.0, 3.0)])
    inner = ScriptedRouter([batch])
    store = FakeCallStore()
    run_id = uuid4()
    router = CachingRouter(inner, store, run_id)

    result = drive(SingleShotLLM(router, max_evaluations=3), make_state(max_evaluations=3))

    assert result.episodes_run == 3
    assert inner.calls == 1, "one real model call for the whole run"
    assert router.hits == 2, "the other two episodes came from the record"


def test_single_shot_exhausts_when_the_model_returns_a_short_batch():
    """Asked for 5, given 2. Not an error — the plan has genuinely run out."""
    batch = BatchProposal(proposals=[proposal(1.0, 1.0), proposal(2.0, 2.0)])
    store = FakeCallStore()
    router = CachingRouter(ScriptedRouter([batch]), store, uuid4())

    result = drive(SingleShotLLM(router, max_evaluations=5), make_state(max_evaluations=5))

    assert result.reason is TerminationReason.EXHAUSTED
    assert result.episodes_run == 2


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class FakeCallStore:
    """The CallStore protocol, backed by a dictionary."""

    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, str], object] = {}

    def record_llm_call(self, run_id, *, call_key, role, provider, model, request, response,
                        prompt_tokens, completion_tokens):
        from sil_agent.persistence.repo import StoredLLMCall

        self.rows.setdefault(
            (run_id, call_key),
            StoredLLMCall(
                call_key=call_key,
                role=role,
                provider=provider,
                model=model,
                request=request,
                response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    def load_llm_call(self, run_id, call_key):
        return self.rows.get((run_id, call_key))  # type: ignore[return-value]


def test_an_identical_prompt_is_answered_from_the_record():
    inner = ScriptedRouter([proposal(1.0, 2.0)])
    router = CachingRouter(inner, FakeCallStore(), uuid4())
    prompt = Prompt(system="s", user="u", template_version="planner.v1")

    first, _ = router.complete(Role.PLANNER, prompt, PlannerProposal)
    second, cost = router.complete(Role.PLANNER, prompt, PlannerProposal)

    assert inner.calls == 1
    assert first.params == second.params
    assert cost.calls == 0, "a replay spends nothing"


def test_offline_replay_refuses_to_call_a_provider():
    """The guarantee, not the hope.

    A replayed run either reproduces from the record or fails loudly. It must
    never quietly make a fresh call that costs money and returns something
    different.
    """
    router = CachingRouter(ScriptedRouter([proposal(1.0, 1.0)]), FakeCallStore(), uuid4(),
                           offline=True)

    with pytest.raises(LookupError, match="offline replay"):
        router.complete(Role.PLANNER, Prompt(system="s", user="u"), PlannerProposal)


def test_the_cache_key_changes_with_the_prompt_version():
    """Bumping a prompt to v2 must not serve v1's answers.

    The whole point of versioning prompts is that a different prompt is a
    different question.
    """
    v1 = Prompt(system="s", user="u", template_version="planner.v1")
    v2 = Prompt(system="s", user="u", template_version="planner.v2")

    assert call_key(Role.PLANNER, v1) != call_key(Role.PLANNER, v2)


def test_the_cache_key_changes_with_the_role():
    prompt = Prompt(system="s", user="u")
    assert call_key(Role.PLANNER, prompt) != call_key(Role.CRITIC, prompt)
