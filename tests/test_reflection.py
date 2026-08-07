"""The critic and the replanner, driven through the real loop with a fake model.

Same discipline as `test_llm_strategies.py`: the loop, the guard, the budget
accounting and the persistence are real, and only the model is scripted. An
episode here makes three model calls instead of one, so the router dispatches by
role.

The tests that matter most are the ones asserting what the model *cannot* do.
Rule 2 is enforced by the shape of `CriticVerdict` rather than by asking the
model nicely, and a schema is only a guarantee for as long as someone is
checking that the fields stayed out of it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from uuid import uuid4

import pytest
from pydantic import BaseModel

from sil_agent.agent.critic import (
    CRITIC_UNAVAILABLE,
    CriticVerdict,
    describe_computed,
    describe_outcome,
    evaluation_from,
    render_critic_prompt,
)
from sil_agent.agent.loop import run_loop
from sil_agent.agent.planner import PlannerProposal, describe_reflection, render_planner_prompt
from sil_agent.agent.replanner import REPLANNER_UNAVAILABLE, ReplannerChoice
from sil_agent.agent.state import (
    BudgetState,
    Candidate,
    CandidateSource,
    CostRecord,
    Episode,
    Evaluation,
    ReplanAction,
    ReplanDecision,
    RunState,
    RunStatus,
    SimResult,
    TerminationReason,
    ToolError,
    utcnow,
)
from sil_agent.prompts import load
from sil_agent.services.router import LLMOutputError, Prompt, Role
from sil_agent.simulators.toy import ToySimulator
from sil_agent.strategies.base import Reflects
from sil_agent.strategies.llm_agent import (
    NO_REFLECTION_BLOCK,
    AgentFull,
    AgentNoReflection,
    AgentPromptControl,
)
from tests.test_loop_termination import InMemoryRepository

GOAL = ToySimulator.from_name("branin").default_goal()


# ---------------------------------------------------------------------------
# A router that answers by role
# ---------------------------------------------------------------------------


class RoleRouter:
    """A ModelRouter that scripts each role separately.

    `ScriptedRouter` in `test_llm_strategies.py` pops from one list, which was
    enough while an episode was one call. A reflecting episode asks three
    different questions and the answers have to be matched to them.
    """

    def __init__(self, *, planner=None, critic=None, replanner=None) -> None:
        self.scripts: dict[Role, list] = {
            Role.PLANNER: list(planner or []),
            Role.CRITIC: list(critic or []),
            Role.REPLANNER: list(replanner or []),
        }
        self.prompts: dict[Role, list[Prompt]] = defaultdict(list)
        self.calls: Counter[Role] = Counter()
        self._fallbacks = 0

    def complete(self, role: Role, prompt: Prompt, schema: type[BaseModel]):
        self.calls[role] += 1
        self.prompts[role].append(prompt)

        script = self.scripts[role]
        reply = script.pop(0) if script else self._default(role)
        if isinstance(reply, Exception):
            raise reply

        return reply, CostRecord(
            calls=1,
            prompt_tokens=100,
            completion_tokens=20,
            model="fake:m",
            requests=1,
            compliant_requests=1,
        )

    def _default(self, role: Role) -> BaseModel:
        if role is Role.PLANNER:
            # Varying, so the duplicate guard does not perturb every proposal
            # and make the assertions about params harder to read than the
            # behaviour they are checking.
            self._fallbacks += 1
            return PlannerProposal(params={"x1": float(self._fallbacks), "x2": 3.0})
        if role is Role.CRITIC:
            return CriticVerdict(diagnosis="a default diagnosis", confidence=0.4)
        return ReplannerChoice(action="EXPLORE", reason="default")

    @property
    def total(self) -> int:
        return sum(self.calls.values())


def verdict(text: str = "objective rose when x1 moved right", confidence: float = 0.6):
    return CriticVerdict(diagnosis=text, hypotheses=["lower x1"], confidence=confidence)


def make_state(*, max_evaluations: int) -> RunState:
    now = utcnow()
    return RunState(
        run_id=uuid4(),
        goal=GOAL,
        status=RunStatus.PENDING,
        history=[],
        best=None,
        budget=BudgetState(max_evaluations=max_evaluations),
        step_idx=0,
        seed=1,
        created_at=now,
        updated_at=now,
    )


def drive(strategy, state, repo=None, **kwargs):
    return run_loop(
        state=state,
        simulator=ToySimulator.from_name("branin"),
        strategy=strategy,
        repo=repo or InMemoryRepository(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Rule 2, enforced by the type rather than by the prompt
# ---------------------------------------------------------------------------


def test_the_critic_cannot_return_a_computed_field():
    """The whole Phase 4 design decision, as one assertion.

    `improved`, `delta_vs_best` and `feasible` are computed from the simulator.
    If they were fields on the schema the model fills in, Rule 2 would be a
    convention the prompt asks for. They are not on it, so there is no channel
    through which a model's opinion could reach them — and deleting this test is
    the only way that could change without anyone noticing.
    """
    fields = set(CriticVerdict.model_fields)

    assert fields == {"diagnosis", "hypotheses", "confidence"}
    assert not fields & {"improved", "delta_vs_best", "feasible"}


def test_evaluation_is_assembled_from_the_oracle_and_the_model():
    computed = Evaluation.computed_only(improved=True, delta_vs_best=1.25, feasible=True)

    joined = evaluation_from(computed, verdict("because x2 was too high", confidence=0.9))

    # From the oracle.
    assert joined.improved is True
    assert joined.delta_vs_best == 1.25
    assert joined.feasible is True
    # From the model.
    assert joined.diagnosis == "because x2 was too high"
    assert joined.confidence == 0.9
    assert joined.hypotheses == ["lower x1"]


def test_a_critic_insisting_it_improved_does_not_change_the_record():
    """The prose is free to be wrong. The flag is not the prose."""
    router = RoleRouter(
        planner=[PlannerProposal(params={"x1": 8.0, "x2": 14.0})],
        critic=[verdict("this is a clear improvement on everything so far", confidence=1.0)],
        replanner=[ReplannerChoice(action="EXPLOIT")],
    )
    repo = InMemoryRepository()
    state = make_state(max_evaluations=1)

    result = drive(AgentFull(router, max_evaluations=1), state, repo)

    stored = repo.episodes[(result.state.run_id, 0)]
    # First episode, so it did improve on "nothing" — the point is that the flag
    # comes from `SimResult.better_than`, and the diagnosis is stored beside it
    # rather than instead of it.
    assert stored.evaluation.improved is True
    assert "clear improvement" in stored.evaluation.diagnosis
    assert stored.evaluation.delta_vs_best == 0.0, "no incumbent to improve on yet"


def test_confidence_outside_zero_to_one_is_rejected_by_the_schema():
    """A model answering 95 for "95%" would otherwise poison ConfidenceDecline."""
    with pytest.raises(ValueError):
        CriticVerdict(diagnosis="d", confidence=95.0)


# ---------------------------------------------------------------------------
# The loop, reflecting
# ---------------------------------------------------------------------------


def test_a_reflecting_episode_makes_three_calls_and_stores_both_halves():
    router = RoleRouter(
        critic=[verdict("x1 near the upper bound drives the objective up")],
        replanner=[ReplannerChoice(action="EXPLOIT", reason="best region found",
                                   next_focus=["x1"])],
    )
    repo = InMemoryRepository()

    result = drive(AgentFull(router, max_evaluations=1), make_state(max_evaluations=1), repo)

    assert router.calls[Role.PLANNER] == 1
    assert router.calls[Role.CRITIC] == 1
    assert router.calls[Role.REPLANNER] == 1

    stored = repo.episodes[(result.state.run_id, 0)]
    assert stored.evaluation.diagnosis.startswith("x1 near the upper bound")
    assert stored.decision.action is ReplanAction.EXPLOIT
    assert stored.decision.next_focus == ["x1"]


def test_a_non_reflecting_strategy_is_untouched():
    """Every strategy through Phase 3.5 behaved this way and still must.

    The loop asks `isinstance(strategy, Reflects)` and nothing else, so a
    baseline gets the computed evaluation and a placeholder decision exactly as
    before — no critic, no extra call, no change to its numbers.
    """
    router = RoleRouter()
    repo = InMemoryRepository()

    result = drive(
        AgentNoReflection(router, max_evaluations=2), make_state(max_evaluations=2), repo
    )

    assert router.calls[Role.CRITIC] == 0
    assert router.calls[Role.REPLANNER] == 0

    stored = repo.episodes[(result.state.run_id, 0)]
    assert stored.evaluation.diagnosis == ""
    assert stored.decision == ReplanDecision.placeholder()


def test_the_cost_of_all_three_calls_lands_on_the_episode():
    """Otherwise the report would price a reflecting run as a planner-only one."""
    router = RoleRouter()
    repo = InMemoryRepository()

    result = drive(AgentFull(router, max_evaluations=1), make_state(max_evaluations=1), repo)

    cost = repo.episodes[(result.state.run_id, 0)].cost
    assert cost.calls == 3
    assert cost.model_requests == 3
    assert cost.compliant_model_requests == 3
    assert cost.prompt_tokens == 300


# ---------------------------------------------------------------------------
# When reflection fails
# ---------------------------------------------------------------------------


def test_a_failed_critic_still_writes_the_episode():
    """The simulation is already paid for. Losing it to a failed narration
    would be the most expensive possible response to that failure."""
    router = RoleRouter(critic=[LLMOutputError("diagnosis: field required")])
    repo = InMemoryRepository()

    result = drive(AgentFull(router, max_evaluations=1), make_state(max_evaluations=1), repo)

    assert result.reason is TerminationReason.BUDGET
    stored = repo.episodes[(result.state.run_id, 0)]
    assert stored.sim_result is not None, "the result survived"
    assert stored.evaluation.diagnosis.startswith(CRITIC_UNAVAILABLE)
    assert router.calls[Role.REPLANNER] == 0, "no diagnosis means nothing to decide over"


def test_a_failed_replanner_keeps_the_diagnosis():
    router = RoleRouter(
        critic=[verdict("a real diagnosis")],
        replanner=[LLMOutputError("action: field required")],
    )
    repo = InMemoryRepository()

    result = drive(AgentFull(router, max_evaluations=1), make_state(max_evaluations=1), repo)

    stored = repo.episodes[(result.state.run_id, 0)]
    assert stored.evaluation.diagnosis == "a real diagnosis"
    assert stored.decision.reason.startswith(REPLANNER_UNAVAILABLE)
    assert stored.decision.action is ReplanAction.EXPLORE


def test_a_reflector_that_raises_does_not_lose_the_episode():
    """The loop's backstop, for a bug rather than a model failure.

    A well-behaved reflector reports failures through `Reflection.failure`. This
    covers the case where one does not.
    """

    class BrokenReflector:
        name = "broken"

        def propose(self, goal, history, rng):
            return Candidate(params={"x1": 1.0, "x2": 1.0}, source=CandidateSource.PLANNER)

        def reflect(self, goal, history, candidate, outcome, computed, best):
            raise RuntimeError("reflector is broken")

    repo = InMemoryRepository()
    strategy = BrokenReflector()
    assert isinstance(strategy, Reflects)

    result = drive(strategy, make_state(max_evaluations=1), repo)

    stored = repo.episodes[(result.state.run_id, 0)]
    assert stored.sim_result is not None
    assert "reflector is broken" in stored.evaluation.diagnosis
    assert stored.evaluation.improved is True, "the oracle's verdict is unaffected"


# ---------------------------------------------------------------------------
# TERMINATE: recorded, not obeyed
# ---------------------------------------------------------------------------


def test_a_terminate_recommendation_does_not_end_the_run():
    """Rule 2, and the fairness rule the ablation rests on.

    A strategy that talks itself into quitting at evaluation one has not lost
    the same contest the others were in — it has set its own budget. The
    recommendation is stored so the report can count it.
    """
    router = RoleRouter(replanner=[ReplannerChoice(action="TERMINATE", reason="hopeless")] * 4)
    repo = InMemoryRepository()

    result = drive(AgentFull(router, max_evaluations=4), make_state(max_evaluations=4), repo)

    assert result.reason is TerminationReason.BUDGET
    assert result.episodes_run == 4, "every evaluation was spent, as for every other strategy"
    assert all(
        e.decision.action is ReplanAction.TERMINATE for e in repo.episodes.values()
    ), "and the recommendation is on the record"


def test_terminate_is_obeyed_when_explicitly_enabled():
    """The flag exists so the behaviour is a choice rather than an absence."""
    router = RoleRouter(replanner=[ReplannerChoice(action="TERMINATE", reason="hopeless")])
    result = drive(
        AgentFull(router, max_evaluations=4), make_state(max_evaluations=4),
        honour_terminate=True,
    )

    assert result.reason is TerminationReason.STAGNATION
    assert result.episodes_run == 1
    assert "TERMINATE" in result.detail


def test_the_replanner_cannot_choose_an_unimplemented_action():
    """DECOMPOSE and ESCALATE exist on the enum and have nothing behind them."""
    with pytest.raises(ValueError):
        ReplannerChoice(action="ESCALATE")


# ---------------------------------------------------------------------------
# The feedback path: reflection reaches the next prompt, via the database
# ---------------------------------------------------------------------------


def test_the_next_planner_prompt_carries_the_previous_diagnosis():
    """The mechanism the whole phase rests on, end to end."""
    router = RoleRouter(
        critic=[verdict("the objective is dominated by x2 above 10")],
        replanner=[
            ReplannerChoice(action="EXPLOIT", reason="narrow around x2", next_focus=["x2"])
        ],
    )

    drive(AgentFull(router, max_evaluations=2), make_state(max_evaluations=2))

    second = router.prompts[Role.PLANNER][1].user
    assert "dominated by x2" in second, "the diagnosis reached the next proposal"
    assert "EXPLOIT" in second
    assert "narrow around x2" in second


def test_reflection_is_rebuilt_from_history_not_carried_on_the_object():
    """Rule 1: a resumed run must rebuild the prompt the interrupted one sent.

    `describe_reflection` is given episodes and nothing else, so a fresh process
    holding no memory of the run produces the identical block.
    """
    episode = Episode(
        idx=3,
        candidate=Candidate(params={"x1": 1.0, "x2": 2.0}, source=CandidateSource.PLANNER),
        result=SimResult(
            metrics={"objective": 5.0}, objective_value=5.0, feasible=True, wall_time_s=0.0
        ),
        evaluation=Evaluation(
            improved=False,
            delta_vs_best=-2.0,
            feasible=True,
            diagnosis="the region around x1=1 is flat",
            hypotheses=["try x1 above 5"],
            confidence=0.7,
        ),
        decision=ReplanDecision(
            action=ReplanAction.EXPLORE, reason="flat region", next_focus=["x1"]
        ),
        cost=CostRecord.zero(),
        duration_ms=1,
    )

    block = describe_reflection([episode])

    assert "EXPLORE" in block
    assert "the region around x1=1 is flat" in block
    assert "try x1 above 5" in block
    assert "0.70" in block


def test_the_first_proposal_has_no_reflection_to_show():
    block = describe_reflection([])
    assert "first proposal" in block
    assert "EXPLORE" in block


# ---------------------------------------------------------------------------
# Prompt versioning: v1 must not move
# ---------------------------------------------------------------------------


def test_planner_v1_renders_identically_with_and_without_a_reflection_block():
    """The regression guard for the Phase 3.5 replay cache.

    `render_planner_prompt` always passes `reflection_block` now, and only v2
    mentions it. If v1's rendering moved by a single character, every recorded
    Phase 3.5 call would become unreachable — `call_key` is a hash of this text —
    and `agent_no_reflection`, the control in this very experiment, would
    silently change mid-comparison.
    """
    template = load("planner", "v1")

    without = render_planner_prompt(template, GOAL, [], None, max_evaluations=20)
    with_block = render_planner_prompt(
        template, GOAL, [], None, max_evaluations=20,
        reflection_block="THIS MUST NOT APPEAR",
    )

    assert without == with_block


def test_planner_v2_does_render_the_reflection_block():
    _, user = render_planner_prompt(
        load("planner", "v2"), GOAL, [], None, max_evaluations=20,
        reflection_block="A DISTINCTIVE MARKER",
    )
    assert "A DISTINCTIVE MARKER" in user


def test_the_prompt_control_uses_v2_without_any_reflection():
    """The confound control: v2's wording, none of its content.

    If this arm matches `agent_no_reflection`, the rewording is inert and any
    `agent_full` difference is reflection. That is the only thing separating
    "reflection pays" from "a differently worded prompt pays".
    """
    router = RoleRouter()
    control = AgentPromptControl(router, max_evaluations=2)

    assert not isinstance(control, Reflects), "the control must not reflect"

    drive(control, make_state(max_evaluations=2))

    assert router.calls[Role.CRITIC] == 0
    assert router.calls[Role.REPLANNER] == 0
    for prompt in router.prompts[Role.PLANNER]:
        assert prompt.template_version == "planner.v2"
        assert NO_REFLECTION_BLOCK in prompt.user


# ---------------------------------------------------------------------------
# What the critic is shown
# ---------------------------------------------------------------------------


def test_the_computed_verdict_is_shown_as_fact_not_as_a_question():
    text = describe_computed(
        Evaluation.computed_only(improved=False, delta_vs_best=-1.5, feasible=True)
    )

    assert "did NOT improve" in text
    assert "-1.5" in text
    assert "not open to revision" in text


def test_a_failed_episode_is_still_given_to_the_critic():
    """"You invented a parameter" is a mistake the planner can act on."""
    text = describe_outcome(
        ToolError(kind="GuardRejection", message="unknown parameter(s): velocity"), GOAL
    )

    assert "did NOT reach the simulator" in text
    assert "velocity" in text


def test_an_infeasible_result_names_the_violation():
    goal = ToySimulator.from_name("branin_constrained").default_goal()
    simulator = ToySimulator.from_name("branin_constrained")
    outcome = simulator.run({"x1": 9.0, "x2": 9.0})

    text = describe_outcome(outcome, goal)

    assert "INFEASIBLE" in text
    assert "x1_plus_x2" in text


def test_the_critic_prompt_carries_the_candidate_and_the_grade():
    candidate = Candidate(
        params={"x1": 3.0, "x2": 2.0}, rationale="near the last best",
        source=CandidateSource.PLANNER,
    )
    outcome = ToySimulator.from_name("branin").run(candidate.params)
    computed = Evaluation.computed_only(improved=True, delta_vs_best=2.0, feasible=True)

    system, user = render_critic_prompt(
        load("critic", "v1"), GOAL, [], candidate, outcome, computed, None
    )

    assert "x1=3" in user
    assert "near the last best" in user
    assert "IMPROVED" in user
    assert "explain" in f"{system}\n{user}".lower()


# ---------------------------------------------------------------------------
# Cost arithmetic
# ---------------------------------------------------------------------------


def test_costs_add_without_losing_the_request_count():
    planner = CostRecord(calls=1, prompt_tokens=100, completion_tokens=10,
                         requests=1, compliant_requests=1)
    critic = CostRecord(calls=2, prompt_tokens=200, completion_tokens=20,
                        repair_attempts=1, requests=1, compliant_requests=0)

    total = planner.plus(critic)

    assert total.calls == 3, "three provider generations"
    assert total.model_requests == 2, "two questions asked"
    assert total.compliant_model_requests == 1, "one of them needed repairing"
    assert total.prompt_tokens == 300


def test_a_pre_phase_4_cost_record_still_reports_one_request():
    """Episodes written before `requests` existed must keep their old numbers.

    The report is regenerated from the episodes table, so a Phase 3 report has
    to keep producing the figures it originally published.
    """
    old = CostRecord(calls=1, prompt_tokens=50, completion_tokens=5)

    assert old.model_requests == 1
    assert old.compliant_model_requests == 1

    repaired = CostRecord(calls=2, repair_attempts=1)
    assert repaired.model_requests == 1
    assert repaired.compliant_model_requests == 0


def test_a_replay_costs_no_request():
    """Otherwise re-running an experiment would report 100% compliance."""
    replay = CostRecord(calls=0, prompt_tokens=100, completion_tokens=10)
    assert replay.model_requests == 0
    assert replay.compliant_model_requests == 0
