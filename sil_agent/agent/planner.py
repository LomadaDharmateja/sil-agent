"""The planner — the LLM proposes.

This is the first component in the project that asks a model for something the
system then acts on, so it is where Rule 2 stops being a design principle and
starts being code:

    model returns text
      -> router: JSON + schema validation      (one repair attempt)
      -> Candidate construction                (rejects booleans)
      -> validate(candidate, parameter_space)  (the Phase 1 guard, unchanged)
      -> simulator

Nothing skips a step. A parameter the model invented is caught by the guard, not
by the simulator failing strangely three lines later.

**Everything the planner sees comes from persisted state.** It is handed
``(goal, history, best)`` and holds nothing between calls — the same reason the
Phase 2 strategies take an explicit RNG. A planner that accumulated context in
memory would produce different prompts before and after a restart, and the run
would silently stop being reproducible even in structure.

**What the model is shown is a decision, not an accident.** Feeding all fifty
episodes into every prompt makes the last call an order of magnitude more
expensive than the first, and buries the useful evidence in noise. The planner
shows the best few results and the most recent few, which is the usual answer
and is recorded here so it can be changed deliberately.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from sil_agent.agent.state import (
    Best,
    Candidate,
    CandidateSource,
    CostRecord,
    Direction,
    Episode,
    Goal,
    ParameterSpace,
    ParamKind,
)
from sil_agent.prompts import PromptTemplate, load
from sil_agent.services.router import ModelRouter, Prompt, Role

# How much history the planner is shown. Small on purpose — see the module
# docstring.
BEST_SHOWN = 5
RECENT_SHOWN = 5


class PlannerProposal(BaseModel):
    """What the planner is asked to return.

    ``params`` deliberately accepts any key. Restricting it to the declared
    parameter names here would make an invented parameter a schema error inside
    the router, which retries — and a model that hallucinates a parameter should
    be *recorded doing so*, not quietly given another go. The guard rejects it,
    the rejection becomes an episode, and the ablation reports the rate.
    """

    model_config = ConfigDict(extra="forbid")

    params: dict[str, float | int | str]
    rationale: str = Field(default="")


class BatchProposal(BaseModel):
    """Single-shot output: the whole experiment plan in one reply."""

    model_config = ConfigDict(extra="forbid")

    proposals: list[PlannerProposal] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Rendering state into prompt text
# ---------------------------------------------------------------------------


def describe_space(space: ParameterSpace) -> str:
    """The parameter space, as the model should see it.

    This text is the contract the guard later enforces, so it states bounds
    exactly rather than describing them loosely.
    """
    lines = []
    for spec in space.params:
        if spec.kind is ParamKind.CATEGORICAL:
            assert spec.choices is not None
            detail = f"one of {spec.choices}"
        else:
            assert spec.bounds is not None
            low, high = spec.bounds
            kind = "integer" if spec.kind is ParamKind.INT else "float"
            detail = f"{kind} in [{low:g}, {high:g}]"
        unit = f" ({spec.unit})" if spec.unit else ""
        lines.append(f"- {spec.name}: {detail}{unit}")
    return "\n".join(lines)


def describe_objective(goal: Goal) -> str:
    direction = "Minimise" if goal.objective.direction is Direction.MINIMISE else "Maximise"
    text = f"{direction} `{goal.objective.metric}`."
    if goal.objective.target is not None:
        text += f" Target: {goal.objective.target:g} or better."
    if goal.raw_text:
        text += f"\n\n{goal.raw_text}"
    return text


def describe_constraints(goal: Goal) -> str:
    if not goal.constraints:
        return ""
    symbols = {"LE": "<=", "LT": "<", "GE": ">=", "GT": ">", "EQ": "=="}
    lines = [
        f"- {c.metric} {symbols[c.operator.value]} {c.threshold:g}" for c in goal.constraints
    ]
    return "## Constraints\n\nA result violating any of these is infeasible.\n\n" + "\n".join(lines)


def format_params(params: dict[str, float | int | str]) -> str:
    parts = []
    for name, value in params.items():
        parts.append(f"{name}={value:.4g}" if isinstance(value, float) else f"{name}={value}")
    return ", ".join(parts)


def describe_history(
    history: Sequence[Episode],
    goal: Goal,
    *,
    best_shown: int = BEST_SHOWN,
    recent_shown: int = RECENT_SHOWN,
) -> str:
    """The best few results and the most recent few, plus any rejections.

    Rejections are included deliberately. A planner that invented a parameter
    last turn needs to be told, or it will invent the same one again — and that
    failure mode is otherwise invisible to it.
    """
    if not history:
        return "Nothing has been evaluated yet. This is the first proposal."

    evaluated = [e for e in history if e.sim_result is not None]
    rejected = [e for e in history if e.sim_result is None]

    sections: list[str] = []

    if evaluated:
        minimising = goal.objective.direction is Direction.MINIMISE
        ranked = sorted(
            evaluated,
            key=lambda e: e.sim_result.objective_value,  # type: ignore[union-attr]
            reverse=not minimising,
        )
        sections.append(
            "Best results so far:\n"
            + "\n".join(_describe_episode(e) for e in ranked[:best_shown])
        )

        recent = evaluated[-recent_shown:]
        if recent:
            sections.append(
                "Most recent evaluations:\n"
                + "\n".join(_describe_episode(e) for e in recent)
            )

    if rejected:
        # Only the last few: the same mistake repeated ten times needs saying
        # once, and the budget is better spent on results.
        recent_rejections = rejected[-3:]
        sections.append(
            "Proposals that were REJECTED before reaching the simulator "
            "(these cost you an attempt and produced no result):\n"
            + "\n".join(
                f"- {format_params(e.candidate.params)} -> {e.result.message}"  # type: ignore[union-attr]
                for e in recent_rejections
            )
        )

    return "\n\n".join(sections)


def _describe_episode(episode: Episode) -> str:
    result = episode.sim_result
    assert result is not None
    feasibility = "" if result.feasible else "  [INFEASIBLE: "
    if not result.feasible:
        violations = "; ".join(
            f"{v.metric}={v.actual:.4g} violates {v.operator.value} {v.threshold:g}"
            for v in result.constraint_violations
        )
        feasibility = f"  [INFEASIBLE: {violations}]"
    return (
        f"- {format_params(episode.candidate.params)} "
        f"-> {result.objective_value:.6g}{feasibility}"
    )


def describe_best(best: Best | None) -> str:
    if best is None:
        return "No feasible result yet."
    return (
        f"Best so far: {best.result.objective_value:.6g} "
        f"at {format_params(best.candidate.params)} (episode {best.episode_idx})"
    )


def describe_reflection(history: Sequence[Episode]) -> str:
    """The previous episode's diagnosis and decision, as the planner sees them.

    **This is the entire mechanism by which reflection reaches the next
    proposal**, and it reads from ``history`` — that is, out of the episodes
    table — rather than from anything held on a strategy object. Rule 1 is why:
    kill the process after episode 7 and resume it, and episode 8's prompt is
    rebuilt from the database with the reflection still in it. A field on the
    strategy would have restarted empty and the resumed run would silently have
    stopped being the run that was interrupted.

    Only the most recent episode is shown. Diagnoses accumulate quickly and the
    local model runs in a 4096-token context; the older ones are already
    reflected in where the search has been, which the history block shows.
    """
    if not history:
        return (
            "No result has been reviewed yet — this is the first proposal. "
            "Treat this as EXPLORE: sample somewhere that will tell you the most "
            "about the shape of the space."
        )

    last = history[-1]
    evaluation = last.evaluation
    decision = last.decision

    lines = [
        f"Direction for this proposal: **{decision.action.value}**",
    ]
    if decision.reason:
        lines.append(f"Why: {decision.reason}")
    if decision.next_focus:
        lines.append(f"Concentrate on: {', '.join(decision.next_focus)}")

    lines.append("")
    lines.append(
        f"Review of episode {last.idx} "
        f"({'improved' if evaluation.improved else 'did not improve'}, "
        f"confidence {evaluation.confidence:.2f}):"
    )
    lines.append(evaluation.diagnosis or "(no diagnosis available)")

    if evaluation.hypotheses:
        lines.append("")
        lines.append("Hypotheses to test:")
        lines.extend(f"- {item}" for item in evaluation.hypotheses)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


def render_planner_prompt(
    template: PromptTemplate,
    goal: Goal,
    history: Sequence[Episode],
    best: Best | None,
    *,
    max_evaluations: int,
    reflection_block: str = "",
) -> tuple[str, str]:
    """Render the exact (system, user) pair the planner sends.

    Extracted from ``Planner.propose`` so that the benchmark-anonymity test can
    inspect *what the model actually receives*. A test that rebuilt the prompt
    itself would keep passing after this function changed, which is precisely
    the leak it exists to catch.

    ``reflection_block`` is always supplied and only ``planner.v2`` consumes it.
    ``string.Template.substitute`` ignores values the template does not mention,
    so v1 renders byte-identically to how it always has — which matters more
    than it looks: v1 is what ``agent_no_reflection`` uses, it is the control in
    the Phase 4 experiment, and its recorded Phase 3.5 calls are keyed on a hash
    of this exact text. One code path, two templates, no drift.
    """
    evaluated = sum(1 for e in history if e.sim_result is not None)

    system, user = template.render(
        objective=describe_objective(goal),
        parameter_space=describe_space(goal.parameter_space),
        constraints=describe_constraints(goal),
        evaluations_used=evaluated,
        max_evaluations=max_evaluations,
        best_block=describe_best(best),
        history_block=describe_history(history, goal),
        reflection_block=reflection_block,
    )
    return system, user


class Planner:
    """Asks the model for one candidate, given persisted state only."""

    def __init__(self, router: ModelRouter, *, prompt_version: str = "v1") -> None:
        self._router = router
        self._template = load("planner", prompt_version)

    def propose(
        self,
        goal: Goal,
        history: Sequence[Episode],
        best: Best | None,
        *,
        max_evaluations: int,
        reflection_block: str = "",
    ) -> tuple[Candidate, CostRecord]:
        """Propose one candidate. Raises LLMOutputError if the model never complies.

        The returned Candidate has *not* been through the guard — that happens in
        the loop, in exactly the same place it happens for every other strategy.
        Validating here would give the LLM path its own private guard and make
        the ablation compare two different pipelines.
        """
        system, user = render_planner_prompt(
            self._template,
            goal,
            history,
            best,
            max_evaluations=max_evaluations,
            reflection_block=reflection_block,
        )

        proposal, cost = self._router.complete(
            Role.PLANNER,
            Prompt(system=system, user=user, template_version=self._template.identifier),
            PlannerProposal,
        )

        return (
            Candidate(
                params=proposal.params,
                rationale=proposal.rationale,
                source=CandidateSource.PLANNER,
            ),
            cost,
        )
