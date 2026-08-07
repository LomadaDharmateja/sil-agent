"""The replanner — a structured choice over the critic's output.

The critic says what happened and why. The replanner says what to do about it,
as one of a small set of named modes plus a focus for the next proposal. Both
are carried into the next planner prompt out of the episodes table, so the whole
mechanism is a pure function of persisted state.

**Two deliberate narrowings, both about not offering a model something the
system cannot honour.**

*The action set is four, not six.* ``ReplanAction`` defines six values.
``DECOMPOSE`` and ``ESCALATE`` have no implementation behind them — escalation
is `TECHNICAL_DESIGN.md` §11.5 and lands in Phase 11 — and a model offered an
action nothing will act on picks it eventually, producing an episode whose
recorded decision is a fiction. So the schema offers what exists.

*``TERMINATE`` is recorded and not obeyed.* The replanner may recommend
stopping; the loop does not stop. That is Rule 2 — deterministic code decides
termination — and it is also the fairness rule the whole ablation rests on. A
strategy that talks itself into quitting at evaluation six has not lost the same
contest the others were in; it has set its own budget. The recommendation rate
is reported instead, which is strictly more informative than letting it happen,
because it is a number rather than a confound. See ``agent/loop.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sil_agent.agent.critic import (
    CRITIC_BEST_SHOWN,
    CRITIC_RECENT_SHOWN,
)
from sil_agent.agent.planner import (
    describe_best,
    describe_history,
    describe_objective,
    describe_space,
)
from sil_agent.agent.state import (
    Best,
    CostRecord,
    Episode,
    Evaluation,
    Goal,
    ReplanAction,
    ReplanDecision,
)
from sil_agent.prompts import PromptTemplate, load
from sil_agent.services.router import ModelRouter, Prompt, Role

# Keep the steering signal short. `next_focus` is echoed into the next planner
# prompt, so an unbounded list is context spent on the replanner's enthusiasm
# rather than on results.
MAX_FOCUS = 4

# The actions the system can actually carry out. See the module docstring.
OFFERED_ACTIONS = ("EXPLOIT", "EXPLORE", "REPAIR", "TERMINATE")

# How a failed replanner is recorded in the decision's reason. See
# ``CRITIC_UNAVAILABLE`` for why this is a constant.
REPLANNER_UNAVAILABLE = "replanner unavailable"


class ReplannerChoice(BaseModel):
    """What the replanner is asked for.

    All three fields are the model's, and that is correct: unlike the critic's
    ``improved``, none of these is a fact the simulator already established.
    They are advice, recorded as advice, and the loop is free to ignore the one
    piece of it that would change the run's length.
    """

    model_config = ConfigDict(extra="forbid")

    # A Literal rather than the full enum, so the grammar Ollama compiles from
    # this schema physically cannot emit DECOMPOSE or ESCALATE.
    action: Literal["EXPLOIT", "EXPLORE", "REPAIR", "TERMINATE"]
    reason: str = Field(default="")
    next_focus: list[str] = Field(default_factory=list)


def describe_evaluation(evaluation: Evaluation) -> str:
    """The critic's output plus the oracle's, as the replanner sees it."""
    lines = [
        f"Improved: {'yes' if evaluation.improved else 'no'}",
        f"Change against previous best: {evaluation.delta_vs_best:+.6g} "
        "(positive means better)",
        f"Feasible: {'yes' if evaluation.feasible else 'no'}",
        f"Critic confidence: {evaluation.confidence:.2f}",
        "",
        f"Diagnosis: {evaluation.diagnosis or '(none)'}",
    ]
    if evaluation.hypotheses:
        lines.append("")
        lines.append("Hypotheses the critic offered:")
        lines.extend(f"- {item}" for item in evaluation.hypotheses)
    return "\n".join(lines)


def describe_recent_actions(history: Sequence[Episode], *, window: int = 5) -> str:
    """What the replanner has already decided lately.

    Included so that a replanner which has said EXPLORE four times running can
    see that it has. Without it the decision is made fresh every episode from a
    single result, and the mode oscillates — which looks like reasoning and is
    closer to a coin flip.
    """
    if not history:
        return "No decisions yet."
    recent = history[-window:]
    return "\n".join(
        f"- episode {episode.idx}: {episode.decision.action.value}"
        f"{f' ({episode.decision.reason})' if episode.decision.reason else ''}"
        for episode in recent
    )


def render_replanner_prompt(
    template: PromptTemplate,
    goal: Goal,
    history: Sequence[Episode],
    evaluation: Evaluation,
    best: Best | None,
    *,
    evaluations_used: int,
    max_evaluations: int,
) -> tuple[str, str]:
    """Render the exact (system, user) pair the replanner sends."""
    return template.render(
        objective=describe_objective(goal),
        parameter_space=describe_space(goal.parameter_space),
        evaluation_block=describe_evaluation(evaluation),
        best_block=describe_best(best),
        evaluations_used=evaluations_used,
        max_evaluations=max_evaluations,
        remaining=max(0, max_evaluations - evaluations_used),
        recent_actions=describe_recent_actions(history),
        history_block=describe_history(
            history,
            goal,
            best_shown=CRITIC_BEST_SHOWN,
            recent_shown=CRITIC_RECENT_SHOWN,
        ),
    )


class Replanner:
    """Turns a diagnosis into a named next mode."""

    def __init__(self, router: ModelRouter, *, prompt_version: str = "v1") -> None:
        self._router = router
        self._template = load("replanner", prompt_version)

    def decide(
        self,
        goal: Goal,
        history: Sequence[Episode],
        evaluation: Evaluation,
        best: Best | None,
        *,
        max_evaluations: int,
    ) -> tuple[ReplanDecision, CostRecord]:
        """Choose the next mode. Raises ProviderError if the model never complies.

        Routed at ``Role.REPLANNER``, which `TECHNICAL_DESIGN.md` §5 maps to the
        MID tier: a structured choice over the critic's output rather than the
        reasoning-heavy call. With a single local model every tier resolves to
        the same weights, so this is currently wiring rather than a saving — but
        it is the wiring that makes swapping in a cheaper model for this role a
        configuration change, which is the entire reason the router exists.
        """
        evaluations_used = sum(1 for e in history if e.sim_result is not None)

        system, user = render_replanner_prompt(
            self._template,
            goal,
            history,
            evaluation,
            best,
            evaluations_used=evaluations_used,
            max_evaluations=max_evaluations,
        )

        choice, cost = self._router.complete(
            Role.REPLANNER,
            Prompt(system=system, user=user, template_version=self._template.identifier),
            ReplannerChoice,
        )

        return (
            ReplanDecision(
                action=ReplanAction(choice.action),
                reason=choice.reason,
                next_focus=choice.next_focus[:MAX_FOCUS],
            ),
            cost,
        )
