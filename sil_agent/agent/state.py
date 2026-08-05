"""Domain models — the vocabulary of the whole system.

Every model here is a Pydantic v2 model with two settings applied throughout:

``frozen=True``
    Instances cannot be mutated after construction. To "change" a run you build
    a new one. This enforces Rule 1 from ``CLAUDE.md`` mechanically: if nothing
    can be quietly mutated in memory, the only way state moves forward is
    through something that gets persisted.

``extra="forbid"``
    An unexpected field is an error rather than being silently dropped. In
    Phase 1 nothing produces unexpected fields; from Phase 3 the LLM will, and
    this turns "the model invented a key" from a silent no-op into a loud
    failure at the boundary.

Two deliberate deviations from ``docs/TECHNICAL_DESIGN.md`` §2, both explained
at the relevant class: :class:`Best` (instead of ``best: Candidate | None``) and
the extra ``BASELINE`` value on :class:`CandidateSource`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    """Timezone-aware UTC now.

    Naive datetimes are a persistent source of off-by-hours bugs once rows go
    into Postgres and come back out, so the codebase never produces one.
    """
    return datetime.now(UTC)


class Frozen(BaseModel):
    """Base for every domain model. See the module docstring for the settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Enumerations
#
# All are StrEnum so that a value serialises to a readable string in JSONB —
# `"MINIMISE"` in the database rather than `0`. Worth it the first time you
# debug by reading rows straight out of Postgres.
# ---------------------------------------------------------------------------


class Direction(StrEnum):
    MINIMISE = "MINIMISE"
    MAXIMISE = "MAXIMISE"


class ParamKind(StrEnum):
    FLOAT = "FLOAT"
    INT = "INT"
    CATEGORICAL = "CATEGORICAL"


class ConstraintOp(StrEnum):
    LE = "LE"  # <=
    LT = "LT"  # <
    GE = "GE"  # >=
    GT = "GT"  # >
    EQ = "EQ"  # == (within tolerance)


class RunStatus(StrEnum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    CRITIQUING = "CRITIQUING"
    REPLANNING = "REPLANNING"
    DONE = "DONE"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class CandidateSource(StrEnum):
    """Who proposed this candidate.

    ``BASELINE`` is an addition to the list in TECHNICAL_DESIGN §2. The design's
    values all describe an agent's reasoning mode, but Phase 2 runs non-agent
    strategies (random, grid, TPE) through the identical harness, and labelling
    a uniform random draw as ``PLANNER`` would corrupt the ablation analysis.
    """

    PLANNER = "PLANNER"
    EXPLOIT = "EXPLOIT"
    EXPLORE = "EXPLORE"
    REPAIR = "REPAIR"
    PERTURB = "PERTURB"
    SURROGATE = "SURROGATE"
    BASELINE = "BASELINE"


class ReplanAction(StrEnum):
    EXPLOIT = "EXPLOIT"
    EXPLORE = "EXPLORE"
    REPAIR = "REPAIR"
    DECOMPOSE = "DECOMPOSE"
    ESCALATE = "ESCALATE"
    TERMINATE = "TERMINATE"


class TerminationReason(StrEnum):
    SUCCESS = "SUCCESS"
    BUDGET = "BUDGET"
    STAGNATION = "STAGNATION"  # Phase 3
    ERROR = "ERROR"

    # Phase 2 additions.
    #
    # EXHAUSTED: the strategy has nothing left to propose. Grid search over a
    # 6-dimensional space runs out long before its evaluation budget does, and a
    # run that stops for that reason has to be distinguishable in the report
    # from one that used its whole budget — otherwise the comparison table shows
    # grid search "losing" without saying that it never got to play.
    EXHAUSTED = "EXHAUSTED"

    # REJECTIONS: too many proposals were rejected by the guard. Impossible in
    # Phases 1-2, where the proposers are deterministic samplers. From Phase 3
    # it stops a hallucinating planner from looping forever without ever
    # reaching the simulator.
    REJECTIONS = "REJECTIONS"


# ---------------------------------------------------------------------------
# Parameter space — declared by the SIMULATOR, never by the model
# ---------------------------------------------------------------------------


class ParamSpec(Frozen):
    """One tunable parameter.

    The simulator is the authority on what parameters exist. Everything that
    proposes values is checked against this.
    """

    name: str = Field(min_length=1)
    kind: ParamKind
    bounds: tuple[float, float] | None = None
    choices: list[str] | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.kind is ParamKind.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"{self.name}: CATEGORICAL parameter needs choices")
            if self.bounds is not None:
                raise ValueError(f"{self.name}: CATEGORICAL parameter cannot have bounds")
        else:
            if self.bounds is None:
                raise ValueError(f"{self.name}: {self.kind} parameter needs bounds")
            if self.choices is not None:
                raise ValueError(f"{self.name}: {self.kind} parameter cannot have choices")
            low, high = self.bounds
            if low >= high:
                raise ValueError(f"{self.name}: bounds must be (low, high) with low < high")
        return self


class ParameterSpace(Frozen):
    """The full set of parameters a simulator accepts."""

    params: list[ParamSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_names(self) -> Self:
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        return self

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def spec_for(self, name: str) -> ParamSpec | None:
        for param in self.params:
            if param.name == name:
                return param
        return None


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


class Objective(Frozen):
    """What we are optimising, and in which direction.

    ``target`` is optional. When set, it defines the SUCCESS termination exit:
    the run can stop early because it is good enough rather than because it ran
    out of budget.
    """

    metric: str = Field(min_length=1)
    direction: Direction
    target: float | None = None


class Constraint(Frozen):
    """A hard requirement on some metric, e.g. ``x1 + x2 <= 10``."""

    metric: str = Field(min_length=1)
    operator: ConstraintOp
    threshold: float
    tolerance: float = 1e-9  # only used by EQ

    def is_satisfied(self, value: float) -> bool:
        match self.operator:
            case ConstraintOp.LE:
                return value <= self.threshold
            case ConstraintOp.LT:
                return value < self.threshold
            case ConstraintOp.GE:
                return value >= self.threshold
            case ConstraintOp.GT:
                return value > self.threshold
            case ConstraintOp.EQ:
                return abs(value - self.threshold) <= self.tolerance

    def violation_amount(self, value: float) -> float:
        """How far past the threshold, as a non-negative number. 0.0 if satisfied."""
        if self.is_satisfied(value):
            return 0.0
        return abs(value - self.threshold)


class Goal(Frozen):
    """The problem statement.

    From Phase 3 the LLM parses free text into this shape — but only the
    ``objective`` and ``constraints`` parts. ``parameter_space`` always comes
    from the simulator, which is why a model cannot invent a parameter.
    """

    raw_text: str
    objective: Objective
    constraints: list[Constraint] = Field(default_factory=list)
    parameter_space: ParameterSpace


# ---------------------------------------------------------------------------
# Candidate, result, evaluation
# ---------------------------------------------------------------------------


class Candidate(Frozen):
    """One proposed set of parameter values."""

    params: dict[str, float | int | str]
    rationale: str = ""
    source: CandidateSource

    @field_validator("params", mode="before")
    @classmethod
    def _reject_booleans(cls, value: object) -> object:
        """Reject boolean parameter values before the union can absorb them.

        ``bool`` is a subclass of ``int`` in Python, so a ``True`` arriving for a
        numeric parameter is quietly accepted by ``float | int | str`` and stored
        as ``1.0``. That is exactly the kind of silent repair this project does
        not do: a proposer that offers a boolean for a continuous parameter has
        misunderstood the space, and the misunderstanding should surface.

        This has to run at ``mode="before"`` — by the time the union has
        finished, the boolean is already a float and the evidence is gone.
        """
        if isinstance(value, dict):
            for name, item in value.items():
                if isinstance(item, bool):
                    raise ValueError(
                        f"{name}: boolean {item!r} is not a valid parameter value "
                        "(bool is a subclass of int and would silently become 1)"
                    )
        return value


class Violation(Frozen):
    """A constraint that was not met, with the numbers that show it."""

    metric: str
    operator: ConstraintOp
    threshold: float
    actual: float
    amount: float = Field(ge=0.0)


class SimResult(Frozen):
    """What the simulator returned. The oracle. Never produced by an LLM.

    ``outcome`` is a discriminator: it lets Pydantic tell a stored SimResult
    from a stored ToolError when reading JSON back out of the database, without
    guessing from which fields happen to be present.
    """

    outcome: Literal["result"] = "result"
    metrics: dict[str, float]
    objective_value: float
    constraint_violations: list[Violation] = Field(default_factory=list)
    feasible: bool
    artifacts: dict[str, str] = Field(default_factory=dict)
    wall_time_s: float = Field(ge=0.0)

    @property
    def total_violation(self) -> float:
        return sum(v.amount for v in self.constraint_violations)

    def better_than(self, best: Best | None, objective: Objective) -> bool:
        """Is this result an improvement on the best so far?

        This is the deterministic half of Rule 2. ``improved`` is computed here,
        from simulator output, and later *injected into* the critic — the LLM is
        told whether it improved, it never decides.

        Ordering rules, in priority order:

        1. Anything feasible beats anything infeasible.
        2. Two feasible results compare on objective value, by direction.
        3. Two infeasible results compare on total constraint violation, smaller
           being better — so a run that starts in an infeasible region can still
           make measurable progress towards feasibility.

        Ties are not improvements: the comparison is strict. Without that, a
        deterministic simulator returning the same value twice would keep
        resetting "best" and the stagnation detectors in Phase 3 would never
        fire.
        """
        if best is None:
            return True

        incumbent = best.result
        if self.feasible != incumbent.feasible:
            return self.feasible

        if not self.feasible:
            return self.total_violation < incumbent.total_violation

        if objective.direction is Direction.MINIMISE:
            return self.objective_value < incumbent.objective_value
        return self.objective_value > incumbent.objective_value

    def delta_vs(self, best: Best | None, objective: Objective) -> float:
        """Signed improvement over the incumbent — positive means better.

        Computed, not opinion. Injected into the critic alongside ``improved``.
        """
        if best is None:
            return 0.0
        difference = self.objective_value - best.result.objective_value
        if objective.direction is Direction.MINIMISE:
            return -difference
        return difference


class ToolError(Frozen):
    """A failed episode — the simulator raised, or the guard rejected the candidate.

    Stored in the same column as a SimResult so that failures are part of the
    permanent history rather than being swallowed. From Phase 3 the critic reads
    these and the agent can react to its own broken proposals.
    """

    outcome: Literal["error"] = "error"
    kind: str
    message: str
    retryable: bool = False


# A discriminated union: Pydantic reads the "outcome" field to decide which
# model to build, instead of trying each in turn and hoping.
EpisodeOutcome = Annotated[SimResult | ToolError, Field(discriminator="outcome")]


class Evaluation(Frozen):
    """The critic's output.

    The first three fields are COMPUTED from the simulator and passed *into* the
    critic. The last three are the LLM's, from Phase 4. The model explains; it
    does not grade.
    """

    improved: bool
    delta_vs_best: float
    feasible: bool
    diagnosis: str = ""
    hypotheses: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @classmethod
    def computed_only(cls, *, improved: bool, delta_vs_best: float, feasible: bool) -> Evaluation:
        """Phase 1 placeholder: the computed fields are real, the prose is empty.

        The schema is fixed now so that Phase 4 adds a critic without a
        migration or a change to anything that reads episodes.
        """
        return cls(improved=improved, delta_vs_best=delta_vs_best, feasible=feasible)


class ReplanDecision(Frozen):
    """What to do next. LLM-authored from Phase 4."""

    action: ReplanAction
    reason: str = ""
    next_focus: list[str] = Field(default_factory=list)

    @classmethod
    def placeholder(cls) -> ReplanDecision:
        return cls(action=ReplanAction.EXPLORE, reason="phase-1: no replanner")


class CostRecord(Frozen):
    """LLM spend for one episode. All zeros in Phases 1-2, which have no LLM."""

    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_eur: float = Field(default=0.0, ge=0.0)
    model: str | None = None

    @classmethod
    def zero(cls) -> CostRecord:
        return cls()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class BudgetState(Frozen):
    """Three independent budgets; any one exhausted halts the run.

    Phase 1 only ever moves ``evaluations_used`` — the evaluation budget is what
    ``--episodes N`` sets. The full governor (reserve-then-commit around each
    LLM call) is Phase 6; the shape is defined now so it does not need a
    migration later.

    ``max_rejections`` is a Phase 2 addition and is a *separate* exit rather
    than part of ``exhausted()``. The reason is fairness, and it is worth
    spelling out because it is the kind of thing that silently biases a
    benchmark. The evaluation budget must buy the same number of *simulator
    calls* for every strategy. If guard rejections came out of the same
    allowance, a planner that proposed thirty pieces of nonsense would get
    thirty fewer real evaluations than a well-behaved one, and would then lose
    the comparison partly because of the accounting rather than the search. So
    rejections are counted separately, capped separately, and reported
    separately.
    """

    max_evaluations: int = Field(gt=0)
    evaluations_used: int = Field(default=0, ge=0)
    max_cost_eur: float = Field(default=2.0, gt=0.0)
    cost_eur_used: float = Field(default=0.0, ge=0.0)
    max_wall_clock_s: float = Field(default=1800.0, gt=0.0)
    wall_clock_s_used: float = Field(default=0.0, ge=0.0)
    max_rejections: int = Field(default=50, ge=0)
    rejections_used: int = Field(default=0, ge=0)

    @property
    def remaining_evaluations(self) -> int:
        return max(0, self.max_evaluations - self.evaluations_used)

    def exhausted(self) -> bool:
        """True when the run has spent one of its three real budgets.

        Deliberately does not consider rejections — see the class docstring.
        """
        return (
            self.evaluations_used >= self.max_evaluations
            or self.cost_eur_used >= self.max_cost_eur
            or self.wall_clock_s_used >= self.max_wall_clock_s
        )

    def rejections_exceeded(self) -> bool:
        return self.rejections_used >= self.max_rejections

    def with_evaluation(self, *, wall_time_s: float, cost: CostRecord) -> BudgetState:
        """Return a new budget with one evaluation's usage added."""
        return BudgetState(
            max_evaluations=self.max_evaluations,
            evaluations_used=self.evaluations_used + 1,
            max_cost_eur=self.max_cost_eur,
            cost_eur_used=self.cost_eur_used + cost.cost_eur,
            max_wall_clock_s=self.max_wall_clock_s,
            wall_clock_s_used=self.wall_clock_s_used + wall_time_s,
            max_rejections=self.max_rejections,
            rejections_used=self.rejections_used,
        )

    def with_rejection(self, *, cost: CostRecord) -> BudgetState:
        """Return a new budget with one guard rejection added.

        A rejection still costs money — the LLM was called to produce the
        candidate — so the euro budget moves. The evaluation budget does not,
        because the simulator was never reached.
        """
        return BudgetState(
            max_evaluations=self.max_evaluations,
            evaluations_used=self.evaluations_used,
            max_cost_eur=self.max_cost_eur,
            cost_eur_used=self.cost_eur_used + cost.cost_eur,
            max_wall_clock_s=self.max_wall_clock_s,
            wall_clock_s_used=self.wall_clock_s_used,
            max_rejections=self.max_rejections,
            rejections_used=self.rejections_used + 1,
        )


# ---------------------------------------------------------------------------
# Episode and run state
# ---------------------------------------------------------------------------


class Episode(Frozen):
    """One iteration of the loop. Append-only; never updated once written."""

    idx: int = Field(ge=0)
    candidate: Candidate
    result: EpisodeOutcome
    evaluation: Evaluation
    decision: ReplanDecision
    cost: CostRecord
    duration_ms: int = Field(ge=0)

    @property
    def sim_result(self) -> SimResult | None:
        """The SimResult, or None if this episode failed."""
        return self.result if isinstance(self.result, SimResult) else None


class Best(Frozen):
    """The best candidate so far, with the result that proves it.

    TECHNICAL_DESIGN §2 has ``best: Candidate | None``. A bare Candidate is not
    enough: deciding whether a new result is an improvement needs the incumbent's
    objective value, and under Rule 2 that number must come from the simulator,
    not be recomputed or guessed. Keeping the winning SimResult alongside the
    candidate is what makes ``better_than`` a pure comparison of oracle output.
    """

    episode_idx: int = Field(ge=0)
    candidate: Candidate
    result: SimResult


class RunState(Frozen):
    """Everything needed to continue a run — and nothing else.

    Rule 1 in one sentence: given this object, loaded from the database, the next
    episode is fully determined. There is no companion in-memory object that also
    has to be right.

    ``step_idx`` and ``best`` are stored for cheap inspection, but the loop
    recomputes both from the episodes table on load rather than trusting them.
    See ``sil_agent/agent/loop.py``.
    """

    run_id: UUID
    goal: Goal
    status: RunStatus = RunStatus.PENDING
    history: list[Episode] = Field(default_factory=list)
    best: Best | None = None
    budget: BudgetState
    step_idx: int = Field(default=0, ge=0)
    seed: int
    created_at: datetime
    updated_at: datetime

    def advanced(
        self,
        *,
        episode: Episode,
        best: Best | None,
        budget: BudgetState,
        status: RunStatus,
    ) -> RunState:
        """Return the state that results from completing ``episode``.

        A new object rather than a mutation. ``model_copy(update=...)`` would be
        shorter but skips validation, which defeats the point of the schema.
        """
        return RunState(
            run_id=self.run_id,
            goal=self.goal,
            status=status,
            history=[*self.history, episode],
            best=best,
            budget=budget,
            step_idx=episode.idx + 1,
            seed=self.seed,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def with_status(self, status: RunStatus) -> RunState:
        return RunState(
            run_id=self.run_id,
            goal=self.goal,
            status=status,
            history=self.history,
            best=self.best,
            budget=self.budget,
            step_idx=self.step_idx,
            seed=self.seed,
            created_at=self.created_at,
            updated_at=utcnow(),
        )
