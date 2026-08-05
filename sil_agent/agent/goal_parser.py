"""Turning a goal written in plain language into a validated ``Goal``.

The only place the LLM touches the problem specification
(``TECHNICAL_DESIGN.md`` §2), and the highest-stakes call in the system: it runs
once, and if it is wrong every episode afterwards optimises the wrong thing while
looking entirely healthy. Hence the STRONG tier and the validation below.

**The model never supplies the parameter space.** That always comes from
``simulator.describe()``. The model is asked only which *reported metric* is the
objective and what the constraints are — and even those are checked against the
metrics the simulator actually reports. A hallucinated metric fails here, at
episode 0, rather than at episode 50 as a run that optimised something that does
not exist.

That check is the whole design in miniature: the model proposes an
interpretation, and deterministic code decides whether it is admissible.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from sil_agent.agent.planner import describe_space
from sil_agent.agent.state import (
    Constraint,
    ConstraintOp,
    CostRecord,
    Direction,
    Goal,
    Objective,
    ParameterSpace,
)
from sil_agent.prompts import load
from sil_agent.services.router import ModelRouter, Prompt, Role


class ParsedObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    direction: Direction
    target: float | None = None


class ParsedConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    operator: ConstraintOp
    threshold: float


class ParsedGoal(BaseModel):
    """What the model is asked to return. Not yet trusted."""

    model_config = ConfigDict(extra="forbid")

    objective: ParsedObjective
    constraints: list[ParsedConstraint] = Field(default_factory=list)


class GoalParseError(ValueError):
    """The model's interpretation named something the simulator does not report."""


def validate_goal(
    parsed: ParsedGoal,
    *,
    raw_text: str,
    space: ParameterSpace,
    reported_metrics: Sequence[str],
) -> Goal:
    """Check the model's interpretation against what the simulator actually reports.

    Rejects rather than repairs, for the same reason the guard rejects an unknown
    parameter: a metric the simulator never produces is a misunderstanding of the
    problem, and quietly substituting the nearest match would hide it behind a
    run that looks fine.
    """
    known = set(reported_metrics)

    if parsed.objective.metric not in known:
        raise GoalParseError(
            f"objective metric {parsed.objective.metric!r} is not reported by this simulator. "
            f"Reported metrics: {', '.join(sorted(known))}"
        )

    for constraint in parsed.constraints:
        if constraint.metric not in known:
            raise GoalParseError(
                f"constraint metric {constraint.metric!r} is not reported by this simulator. "
                f"Reported metrics: {', '.join(sorted(known))}"
            )

    return Goal(
        raw_text=raw_text,
        objective=Objective(
            metric=parsed.objective.metric,
            direction=parsed.objective.direction,
            target=parsed.objective.target,
        ),
        constraints=[
            Constraint(
                metric=c.metric,
                operator=c.operator,
                threshold=c.threshold,
            )
            for c in parsed.constraints
        ],
        # Always from the simulator. This is the line that makes it impossible
        # for a model to invent a parameter.
        parameter_space=space,
    )


class GoalParser:
    """Parses free text into a Goal, once per run."""

    def __init__(self, router: ModelRouter, *, prompt_version: str = "v1") -> None:
        self._router = router
        self._template = load("goal_parser", prompt_version)

    def parse(
        self,
        raw_text: str,
        *,
        space: ParameterSpace,
        reported_metrics: Sequence[str],
    ) -> tuple[Goal, CostRecord]:
        """Raises GoalParseError if the interpretation names an unknown metric."""
        system, user = self._template.render(
            goal_text=raw_text,
            parameter_space=describe_space(space),
            metrics="\n".join(f"- {metric}" for metric in reported_metrics),
        )

        parsed, cost = self._router.complete(
            Role.GOAL_PARSER,
            Prompt(system=system, user=user, template_version=self._template.identifier),
            ParsedGoal,
        )

        return (
            validate_goal(
                parsed,
                raw_text=raw_text,
                space=space,
                reported_metrics=reported_metrics,
            ),
            cost,
        )
