"""The critic — the LLM explains a result it is told the grade of.

This is Rule 2 in its most important application, and the design document says
so: *"The first three fields are computed from the oracle and passed into the
critic. The LLM explains; it does not grade."*

The obvious way to build this is to ask the model for an ``Evaluation`` and
trust it to leave ``improved``, ``delta_vs_best`` and ``feasible`` alone.
Instead, **the schema the model is asked for does not contain those fields at
all**:

    CriticVerdict = {diagnosis, hypotheses, confidence}

The loop then builds the ``Evaluation`` itself, taking the computed three from
the simulator and the prose three from the verdict. The model is *shown*
``improved`` and ``delta_vs_best`` in its prompt, as facts it must explain, and
has no channel through which to return a different value. A convention the
prompt asks for can be ignored; a field that does not exist cannot be filled in.

**What the critic is for.** Not scoring — the oracle already scored it. The
critic answers *why*: which parameter moved, in which direction, and what that
implies about the shape of the space. That text is the only thing carried
forward into the next proposal, so its quality is the quality of the loop.
`TECHNICAL_DESIGN.md` §5 puts it plainly: *"Diagnosis quality is the value of
the loop."*
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from sil_agent.agent.planner import (
    describe_best,
    describe_history,
    describe_objective,
    describe_space,
    format_params,
)
from sil_agent.agent.state import (
    Best,
    Candidate,
    CostRecord,
    Episode,
    Evaluation,
    Goal,
    SimResult,
    ToolError,
)
from sil_agent.prompts import PromptTemplate, load
from sil_agent.services.router import ModelRouter, Prompt, Role

# How much history the critic is shown. Smaller than the planner's, on purpose:
# the critic is reasoning about *one* result, and the history is context for
# that result rather than the subject. It also keeps the prompt inside the
# 4096-token context the local model runs with, which the planner prompt plus a
# result plus a diagnosis block would otherwise threaten on a 6-parameter space.
CRITIC_BEST_SHOWN = 3
CRITIC_RECENT_SHOWN = 3

# Bound on how much prose comes back. A diagnosis is one or two sentences of
# analysis; a list of eight hypotheses is a model padding, and every token of it
# is carried into the next planner prompt where the context budget is real.
MAX_HYPOTHESES = 3

# How a failed critic is recorded in the diagnosis field. A constant rather than
# a literal because two places write it and the report reads it: a run whose
# critic was down has to be distinguishable from one that never had a critic,
# and that distinction cannot survive a typo in one of three string literals.
CRITIC_UNAVAILABLE = "critic unavailable"


class CriticVerdict(BaseModel):
    """What the critic is asked for — and deliberately nothing more.

    Note what is absent: ``improved``, ``delta_vs_best`` and ``feasible``. Those
    are computed by ``SimResult.better_than`` and ``delta_vs`` from oracle
    output and injected into the prompt. Their absence here is the enforcement
    mechanism, not an oversight, and adding them "for completeness" would delete
    the guarantee this module exists to provide.
    """

    model_config = ConfigDict(extra="forbid")

    diagnosis: str = Field(default="")
    hypotheses: list[str] = Field(default_factory=list)
    # Constrained at the schema so a model answering "0.95" as a percentage
    # (95) is a validation failure the router can repair, rather than a
    # confidence of 95.0 quietly poisoning the stagnation detector that reads it.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def describe_outcome(outcome: SimResult | ToolError, goal: Goal) -> str:
    """The result the critic has to explain, including a failed one.

    A rejected or failed episode is still worth a diagnosis — arguably more so,
    since "you invented a parameter" is a mistake the planner can act on
    directly. Phase 3 already records these as episodes; this is what makes them
    legible.
    """
    if isinstance(outcome, ToolError):
        return (
            f"The proposal did NOT reach the simulator. It failed with "
            f"{outcome.kind}: {outcome.message}\n\n"
            "No objective value exists for this attempt."
        )

    lines = [f"objective = {outcome.objective_value:.6g}"]

    for name, value in outcome.metrics.items():
        if name != goal.objective.metric:
            lines.append(f"{name} = {value:.6g}")

    if outcome.feasible:
        lines.append("All constraints satisfied (feasible).")
    else:
        lines.append("INFEASIBLE — constraints violated:")
        lines.extend(
            f"  {v.metric} = {v.actual:.6g} violates {v.operator.value} {v.threshold:g} "
            f"(by {v.amount:.6g})"
            for v in outcome.constraint_violations
        )

    return "\n".join(lines)


def describe_computed(computed: Evaluation) -> str:
    """The grade, rendered as fact.

    Phrased as a statement rather than a question on purpose. "Did this improve?"
    invites the model to disagree with the oracle; "This did not improve" gives
    it something to explain. The numbers here are the ones already stored on the
    episode, so what the critic was told is recoverable from the database
    afterwards rather than being a property of a prompt nobody kept.
    """
    verdict = "IMPROVED on the best so far" if computed.improved else "did NOT improve"
    feasibility = "feasible" if computed.feasible else "infeasible"
    return (
        f"This result {verdict}.\n"
        f"Change against the previous best: {computed.delta_vs_best:+.6g} "
        "(positive means better).\n"
        f"The result is {feasibility}.\n\n"
        "These three facts are computed from the simulator. They are not open to "
        "revision — your task is to explain them."
    )


def render_critic_prompt(
    template: PromptTemplate,
    goal: Goal,
    history: Sequence[Episode],
    candidate: Candidate,
    outcome: SimResult | ToolError,
    computed: Evaluation,
    best: Best | None,
) -> tuple[str, str]:
    """Render the exact (system, user) pair the critic sends.

    Extracted for the same reason ``render_planner_prompt`` was: the
    benchmark-anonymity test inspects what the model actually receives, and a
    test that rebuilds the prompt itself keeps passing after this function
    starts sending something else.
    """
    return template.render(
        objective=describe_objective(goal),
        parameter_space=describe_space(goal.parameter_space),
        candidate=format_params(candidate.params),
        rationale=candidate.rationale or "(none given)",
        outcome_block=describe_outcome(outcome, goal),
        computed_block=describe_computed(computed),
        best_block=describe_best(best),
        history_block=describe_history(
            history,
            goal,
            best_shown=CRITIC_BEST_SHOWN,
            recent_shown=CRITIC_RECENT_SHOWN,
        ),
    )


class Critic:
    """Asks the model to explain one result. Never asks it to grade one."""

    def __init__(self, router: ModelRouter, *, prompt_version: str = "v1") -> None:
        self._router = router
        self._template = load("critic", prompt_version)

    def evaluate(
        self,
        goal: Goal,
        history: Sequence[Episode],
        candidate: Candidate,
        outcome: SimResult | ToolError,
        computed: Evaluation,
        best: Best | None,
    ) -> tuple[CriticVerdict, CostRecord]:
        """Diagnose one result. Raises ProviderError if the model never complies.

        ``computed`` carries the oracle's verdict *in*. Nothing comes back out
        that could change it — see ``CriticVerdict``.
        """
        system, user = render_critic_prompt(
            self._template, goal, history, candidate, outcome, computed, best
        )

        verdict, cost = self._router.complete(
            Role.CRITIC,
            Prompt(system=system, user=user, template_version=self._template.identifier),
            CriticVerdict,
        )

        # Truncate rather than reject. An over-long list is the model padding,
        # not misunderstanding the schema, and failing the episode over it would
        # throw away a simulation that has already been paid for.
        if len(verdict.hypotheses) > MAX_HYPOTHESES:
            verdict = CriticVerdict(
                diagnosis=verdict.diagnosis,
                hypotheses=verdict.hypotheses[:MAX_HYPOTHESES],
                confidence=verdict.confidence,
            )

        return verdict, cost


def evaluation_from(computed: Evaluation, verdict: CriticVerdict) -> Evaluation:
    """Build the stored Evaluation: oracle fields from ``computed``, prose from the model.

    The single place the two halves are joined, so there is exactly one line in
    the codebase where it would be possible to let the model's opinion into a
    computed field — and it does not.
    """
    return Evaluation(
        improved=computed.improved,
        delta_vs_best=computed.delta_vs_best,
        feasible=computed.feasible,
        diagnosis=verdict.diagnosis,
        hypotheses=verdict.hypotheses,
        confidence=verdict.confidence,
    )
