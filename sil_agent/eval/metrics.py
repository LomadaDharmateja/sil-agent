"""Scoring a run. Deterministic, database-free, unit-testable.

Everything here takes objects and returns numbers. Nothing opens a connection or
reads a file, so all of it can be tested against hand-computed values without a
Postgres anywhere — which matters, because these are the functions that decide
what the project's headline claim turns out to be.

Three rules are enforced here rather than left to the caller:

**Only feasible results count.** A run's score is the best objective value among
evaluations that satisfied every constraint. Letting an infeasible-but-low value
count would flatter exactly the strategy that ignores constraints, which is the
opposite of what the constrained benchmark exists to measure.

**A run with no feasible result scores ``None``, not a large number.** Standing
in a big float would let it be averaged, and the average would be meaningless.
``None`` forces the report to say "no feasible solution found", which is the
truth.

**Regret is clamped at zero.** ``known_optimum`` is a rounded literal — Branin's
0.397887, Hartmann-6's -3.32237 — so a very good run can land a hair below it
and produce a negative regret. Displaying that would suggest a strategy beat the
global optimum.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from uuid import UUID

from sil_agent.agent.state import Direction, Episode, RunStatus, TerminationReason
from sil_agent.persistence.repo import StoredRun
from sil_agent.simulators.toy import Benchmark

# Analysis results are plain frozen dataclasses rather than Pydantic models.
# The domain models in `agent/state.py` are Pydantic because they cross a
# boundary — they are serialised into JSONB and validated on the way back. These
# never leave the process, so validation would buy nothing and cost noise.


@dataclass(frozen=True)
class RunSummary:
    """One run, reduced to the numbers the comparison table needs."""

    run_id: UUID
    strategy: str
    simulator: str
    seed: int
    termination: TerminationReason | None  # None while the run is unfinished

    evaluations: int
    rejections: int
    max_evaluations: int

    best_objective: float | None  # best FEASIBLE objective value
    regret: float | None  # distance from the known optimum, clamped at 0
    feasible_fraction: float
    wall_clock_s: float

    # Best-so-far regret after each evaluation, padded to `max_evaluations` so
    # curves from different runs can be compared point by point.
    regret_curve: tuple[float | None, ...]

    @property
    def is_complete(self) -> bool:
        return self.termination is not None


def derive_termination(stored: StoredRun) -> TerminationReason | None:
    """Work out why a run stopped, from what is in the database.

    The reason is recomputed rather than stored, for the same reason ``best``
    and ``step_idx`` are: a derived value that is also persisted is a value that
    can disagree with the record. Everything needed is already there.

    Returns ``None`` for a run that has not finished — one interrupted by
    Ctrl-C never reaches its final save, so its status is still EXECUTING. That
    is exactly the case the harness needs to detect in order to resume it, and
    it must not be confused with a strategy that finished early.
    """
    state = stored.state
    if state.status is RunStatus.FAILED:
        return TerminationReason.ERROR
    if state.status is not RunStatus.DONE:
        return None

    budget = state.budget
    if budget.rejections_exceeded():
        return TerminationReason.REJECTIONS
    if budget.exhausted():
        return TerminationReason.BUDGET

    # Finished, but with evaluations still available. Either it hit its target
    # or it ran out of things to propose.
    target = state.goal.objective.target
    if target is not None and state.best is not None and state.best.result.feasible:
        return TerminationReason.SUCCESS
    return TerminationReason.EXHAUSTED


def regret_of(value: float, benchmark: Benchmark) -> float:
    """Distance from the known global optimum. Never negative.

    This is why the benchmarks are Branin, Hartmann-6 and Rosenbrock rather than
    something invented: the true answer is published, so solution quality is an
    absolute distance rather than "better than where we started".
    """
    if benchmark.direction is Direction.MINIMISE:
        raw = value - benchmark.known_optimum
    else:
        raw = benchmark.known_optimum - value
    return max(0.0, raw)


def best_feasible_curve(episodes: Sequence[Episode], benchmark: Benchmark) -> list[float | None]:
    """Best feasible objective value after each *evaluation*, in order.

    One entry per episode that reached the simulator. Guard rejections produce
    no entry: they consumed no evaluation, so including them would misalign the
    curve against the x-axis the budget is measured on.

    Entries are ``None`` until the first feasible result appears.
    """
    curve: list[float | None] = []
    best: float | None = None

    for episode in episodes:
        result = episode.sim_result
        if result is None:
            continue
        if result.feasible:
            value = result.objective_value
            if best is None or _is_better(value, best, benchmark.direction):
                best = value
        curve.append(best)

    return curve


def _is_better(value: float, incumbent: float, direction: Direction) -> bool:
    if direction is Direction.MINIMISE:
        return value < incumbent
    return value > incumbent


def summarise_run(stored: StoredRun, benchmark: Benchmark) -> RunSummary:
    """Reduce one stored run to its scores."""
    episodes = stored.state.history
    budget = stored.state.budget

    curve = best_feasible_curve(episodes, benchmark)
    evaluations = len(curve)
    rejections = len(episodes) - evaluations

    best_objective = curve[-1] if curve else None
    regret = regret_of(best_objective, benchmark) if best_objective is not None else None

    feasible = sum(
        1 for e in episodes if e.sim_result is not None and e.sim_result.feasible
    )
    feasible_fraction = feasible / evaluations if evaluations else 0.0

    # Pad so every curve in a cell has the same length. A run that stopped early
    # because its grid ran out holds its final value: that IS its best-so-far at
    # evaluation 200, since it never found anything better.
    regret_curve: list[float | None] = [
        regret_of(value, benchmark) if value is not None else None for value in curve
    ]
    if regret_curve:
        regret_curve.extend([regret_curve[-1]] * (budget.max_evaluations - len(regret_curve)))
    else:
        regret_curve = [None] * budget.max_evaluations

    return RunSummary(
        run_id=stored.state.run_id,
        strategy=stored.strategy,
        simulator=stored.simulator,
        seed=stored.state.seed,
        termination=derive_termination(stored),
        evaluations=evaluations,
        rejections=rejections,
        max_evaluations=budget.max_evaluations,
        best_objective=best_objective,
        regret=regret,
        feasible_fraction=feasible_fraction,
        wall_clock_s=budget.wall_clock_s_used,
        regret_curve=tuple(regret_curve[: budget.max_evaluations]),
    )


def evaluations_to_threshold(summary: RunSummary, threshold: float) -> int | None:
    """How many evaluations until regret first fell to ``threshold`` or below.

    Budget efficiency rather than final quality — "how fast", not "how good".
    This is the number Phase 7's surrogate has to move in order to justify its
    headline claim of cutting simulator calls at equal solution quality, so it
    is baselined here.

    ``None`` means the threshold was never reached.
    """
    for index, value in enumerate(summary.regret_curve, start=1):
        if value is not None and value <= threshold:
            return index
    return None


# ---------------------------------------------------------------------------
# Aggregating across seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellStats:
    """One cell of the comparison table: a strategy on a benchmark, over seeds."""

    strategy: str
    simulator: str
    n: int
    values: tuple[float, ...]  # regret, one per seed that produced a feasible result
    missing: int  # seeds with no feasible result at all
    mean: float | None
    std: float | None
    median: float | None
    iqr: tuple[float, float] | None
    mean_evaluations: float
    terminations: tuple[str, ...]

    @property
    def label(self) -> str:
        """`mean ± std` for the table, or an honest marker when there is nothing to average."""
        if self.mean is None or self.std is None:
            return "no feasible solution"
        return f"{self.mean:.4g} ± {self.std:.2g}"


def summarise_cell(summaries: Sequence[RunSummary]) -> CellStats:
    """Aggregate the runs of one (strategy, benchmark) pair across seeds."""
    if not summaries:
        raise ValueError("cannot summarise an empty cell")

    strategy = summaries[0].strategy
    simulator = summaries[0].simulator

    values = tuple(s.regret for s in summaries if s.regret is not None)
    missing = len(summaries) - len(values)

    return CellStats(
        strategy=strategy,
        simulator=simulator,
        n=len(summaries),
        values=values,
        missing=missing,
        mean=_mean(values),
        std=_sample_std(values),
        median=median(values) if values else None,
        iqr=_iqr(values),
        mean_evaluations=sum(s.evaluations for s in summaries) / len(summaries),
        terminations=tuple(
            sorted({s.termination.value if s.termination else "INCOMPLETE" for s in summaries})
        ),
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float | None:
    """Sample standard deviation (n-1 denominator).

    ``n-1`` rather than ``n`` because these five seeds are a sample of the
    infinitely many runs the strategy could have produced, not the entire
    population of interest. With n=5 the difference is about 12%, which is large
    enough to matter in a table that will be quoted.
    """
    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _iqr(values: Sequence[float]) -> tuple[float, float] | None:
    """The 25th and 75th percentiles, by linear interpolation."""
    if len(values) < 2:
        return None
    return (percentile(values, 25.0), percentile(values, 75.0))


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile. Small samples make method choice visible.

    Written out rather than taken from numpy so the tests can assert against
    hand-computed values, and so the definition in use is on the page.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# ---------------------------------------------------------------------------
# How much of a difference is real?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseFloor:
    """How much a k-seed mean moves purely because of which seeds were drawn.

    The point of this number. In Phase 4 the project will compare an agent
    against TPE over 5 seeds and want to say "the agent is better". Before that
    claim can mean anything, someone has to know how far apart two 5-seed means
    can be when *nothing* differs but the seeds. This measures exactly that, by
    running each baseline over many more seeds than the protocol requires and
    then repeatedly drawing 5 of them.

    Baselines cost nothing to run, so the measurement is nearly free — and it is
    much easier to set this bar honestly now, before there is a result to hope
    for.
    """

    strategy: str
    simulator: str
    population: int  # seeds actually run
    subsample: int  # k, the protocol's seed count
    observed_mean: float
    low: float  # 5th percentile of the k-seed mean
    high: float  # 95th percentile
    spread: float  # high - low: differences smaller than this prove nothing


def seed_noise_floor(
    values: Sequence[float],
    *,
    strategy: str,
    simulator: str,
    subsample: int = 5,
    iterations: int = 10_000,
    seed: int = 0,
) -> NoiseFloor | None:
    """Bootstrap the spread of a ``subsample``-seed mean out of a larger sample.

    Draws without replacement, because the thing being imitated is running the
    protocol with a different set of five seeds — not resampling five seeds with
    repeats.

    Deterministic: the RNG is seeded, so the reported interval is reproducible.
    """
    if len(values) <= subsample:
        return None

    rng = random.Random(f"noise:{strategy}:{simulator}:{seed}")
    means = [sum(rng.sample(list(values), subsample)) / subsample for _ in range(iterations)]

    low = percentile(means, 5.0)
    high = percentile(means, 95.0)

    return NoiseFloor(
        strategy=strategy,
        simulator=simulator,
        population=len(values),
        subsample=subsample,
        observed_mean=sum(values) / len(values),
        low=low,
        high=high,
        spread=high - low,
    )
