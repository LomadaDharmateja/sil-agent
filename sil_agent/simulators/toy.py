"""Standard optimisation benchmark functions, wrapped as simulators.

These are the analytic test functions the optimisation literature uses. Choosing
them over a bespoke toy problem buys three things:

* **True regret.** The global optimum is known, so solution quality can be
  measured as distance from the real answer, not just "better than where we
  started".
* **Credible baselines.** Optuna handles these natively, so Phase 2's TPE
  comparison is honest rather than a hand-rolled approximation.
* **Comparability.** Published results exist for these functions.

The coefficients below were taken from the Surjanovic & Bingham reference
implementations at sfu.ca rather than from memory. One trap is worth spelling
out: that reference's ``hart6.m`` returns the *rescaled* Hartmann-6,
``-(2.58 + outer) / 1.94``, whose minimum is not -3.32237. The unrescaled form
implemented here is the one with the documented -3.32237 optimum.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from sil_agent.agent.state import (
    Constraint,
    ConstraintOp,
    Direction,
    Goal,
    Objective,
    ParameterSpace,
    ParamKind,
    ParamSpec,
    SimResult,
    Violation,
)
from sil_agent.simulators.base import EvalCost

# ---------------------------------------------------------------------------
# The functions themselves
# ---------------------------------------------------------------------------


def branin(x1: float, x2: float) -> float:
    """Branin-Hoo.

    f(x) = a(x2 - b*x1^2 + c*x1 - r)^2 + s(1 - t)cos(x1) + s

    Domain x1 in [-5, 10], x2 in [0, 15]. Three global minima, all with value
    0.397887 — at (-pi, 12.275), (pi, 2.275) and (3*pi, 2.475).
    """
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1.0 - t) * math.cos(x1)
    return term1 + term2 + s


# Hartmann-6 coefficients. alpha is length 4; A and P are both 4x6.
HARTMANN6_ALPHA: tuple[float, ...] = (1.0, 1.2, 3.0, 3.2)

HARTMANN6_A: tuple[tuple[float, ...], ...] = (
    (10.00, 3.00, 17.00, 3.50, 1.70, 8.00),
    (0.05, 10.00, 17.00, 0.10, 8.00, 14.00),
    (3.00, 3.50, 1.70, 10.00, 17.00, 8.00),
    (17.00, 8.00, 0.05, 10.00, 0.10, 14.00),
)

# Published as integers scaled by 1e-4.
HARTMANN6_P: tuple[tuple[float, ...], ...] = tuple(
    tuple(value * 1e-4 for value in row)
    for row in (
        (1312, 1696, 5569, 124, 8283, 5886),
        (2329, 4135, 8307, 3736, 1004, 9991),
        (2348, 1451, 3522, 2883, 3047, 6650),
        (4047, 8828, 8732, 5743, 1091, 381),
    )
)


def hartmann6(x: Sequence[float]) -> float:
    """Hartmann-6, unrescaled.

    f(x) = -sum_i alpha_i * exp(-sum_j A_ij (x_j - P_ij)^2)

    Domain [0, 1]^6. Global minimum approximately -3.32237 at
    (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573).
    """
    if len(x) != 6:
        raise ValueError(f"hartmann6 needs 6 values, got {len(x)}")

    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j in range(6):
            inner += HARTMANN6_A[i][j] * (x[j] - HARTMANN6_P[i][j]) ** 2
        outer += HARTMANN6_ALPHA[i] * math.exp(-inner)
    return -outer


def rosenbrock(x: Sequence[float]) -> float:
    """Rosenbrock's valley.

    f(x) = sum_{i=1}^{d-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]

    Global minimum exactly 0 at x_i = 1 for all i. Easy to find the valley,
    hard to follow it to the bottom, which makes it a good discriminator
    between strategies that exploit well and strategies that only explore.
    """
    if len(x) < 2:
        raise ValueError(f"rosenbrock needs at least 2 values, got {len(x)}")

    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1.0) ** 2
    return total


ROSENBROCK_DIM = 4


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


def _space(names: Sequence[str], bounds: Sequence[tuple[float, float]]) -> ParameterSpace:
    return ParameterSpace(
        params=[
            ParamSpec(name=name, kind=ParamKind.FLOAT, bounds=bound)
            for name, bound in zip(names, bounds, strict=True)
        ]
    )


@dataclass(frozen=True)
class Benchmark:
    """A benchmark function plus everything needed to optimise and score it.

    ``known_optimum`` and ``known_optimisers`` are not used by the loop. They
    exist so that Phase 2 can report true regret (how far from the real answer)
    rather than only relative improvement, and so the tests here can assert the
    implementation is correct rather than merely self-consistent.
    """

    name: str
    space: ParameterSpace
    objective_metric: str
    direction: Direction
    objective: Callable[[list[float]], float]
    known_optimum: float
    known_optimisers: tuple[tuple[float, ...], ...]
    goal_text: str
    constraints: tuple[Constraint, ...] = ()
    extra_metrics: Callable[[list[float]], dict[str, float]] = field(default=lambda _: {})


BRANIN = Benchmark(
    name="branin",
    space=_space(["x1", "x2"], [(-5.0, 10.0), (0.0, 15.0)]),
    objective_metric="branin",
    direction=Direction.MINIMISE,
    objective=lambda x: branin(x[0], x[1]),
    known_optimum=0.397887,
    known_optimisers=(
        (-math.pi, 12.275),
        (math.pi, 2.275),
        (3.0 * math.pi, 2.475),
    ),
    goal_text="Minimise the Branin function over x1 in [-5, 10] and x2 in [0, 15].",
)

# The constrained variant exists so that constraint handling — violations,
# feasibility, and "feasible beats infeasible" ordering — is exercised from
# episode 0 rather than first appearing in Phase 9 with a real simulator.
BRANIN_CONSTRAINED = Benchmark(
    name="branin_constrained",
    space=BRANIN.space,
    objective_metric="branin",
    direction=Direction.MINIMISE,
    objective=lambda x: branin(x[0], x[1]),
    known_optimum=0.397887,
    # Two of Branin's three global minima survive x1 + x2 <= 10:
    #   (-pi, 12.275) sums to  9.13  -> feasible
    #   ( pi,  2.275) sums to  5.42  -> feasible
    #   (3pi,  2.475) sums to 11.90  -> excluded by the constraint
    # So the constrained optimum value is unchanged at 0.397887.
    known_optimisers=((-math.pi, 12.275), (math.pi, 2.275)),
    goal_text=(
        "Minimise the Branin function subject to x1 + x2 <= 10, "
        "with x1 in [-5, 10] and x2 in [0, 15]."
    ),
    constraints=(Constraint(metric="x1_plus_x2", operator=ConstraintOp.LE, threshold=10.0),),
    extra_metrics=lambda x: {"x1_plus_x2": x[0] + x[1]},
)

HARTMANN6 = Benchmark(
    name="hartmann6",
    space=_space([f"x{i}" for i in range(1, 7)], [(0.0, 1.0)] * 6),
    objective_metric="hartmann6",
    direction=Direction.MINIMISE,
    objective=hartmann6,
    known_optimum=-3.32237,
    known_optimisers=((0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),),
    goal_text="Minimise the 6-dimensional Hartmann function over the unit hypercube.",
)

ROSENBROCK = Benchmark(
    name="rosenbrock",
    space=_space(
        [f"x{i}" for i in range(1, ROSENBROCK_DIM + 1)],
        [(-5.0, 10.0)] * ROSENBROCK_DIM,
    ),
    objective_metric="rosenbrock",
    direction=Direction.MINIMISE,
    objective=rosenbrock,
    known_optimum=0.0,
    known_optimisers=((1.0,) * ROSENBROCK_DIM,),
    goal_text=f"Minimise the {ROSENBROCK_DIM}-dimensional Rosenbrock function over [-5, 10].",
)

BENCHMARKS: dict[str, Benchmark] = {
    benchmark.name: benchmark for benchmark in (BRANIN, BRANIN_CONSTRAINED, HARTMANN6, ROSENBROCK)
}


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


class ToySimulator:
    """Analytic, instant, deterministic. Develop and debug against this.

    Satisfies the ``Simulator`` protocol structurally — it does not inherit from
    anything.
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self._benchmark = benchmark

    @classmethod
    def from_name(cls, name: str) -> ToySimulator:
        if name not in BENCHMARKS:
            available = ", ".join(sorted(BENCHMARKS))
            raise KeyError(f"unknown benchmark {name!r}; available: {available}")
        return cls(BENCHMARKS[name])

    @property
    def name(self) -> str:
        return self._benchmark.name

    @property
    def benchmark(self) -> Benchmark:
        return self._benchmark

    def describe(self) -> ParameterSpace:
        return self._benchmark.space

    @property
    def cost_per_eval(self) -> EvalCost:
        return EvalCost(seconds=0.0, euros=0.0)

    def default_goal(self) -> Goal:
        """The goal this benchmark implies.

        Phase 1 has no LLM, so the goal is built deterministically from the
        benchmark definition. From Phase 3 an LLM parses free text into this same
        shape — but only the objective and constraints. ``parameter_space``
        always comes from ``describe()``.
        """
        return Goal(
            raw_text=self._benchmark.goal_text,
            objective=Objective(
                metric=self._benchmark.objective_metric,
                direction=self._benchmark.direction,
            ),
            constraints=list(self._benchmark.constraints),
            parameter_space=self._benchmark.space,
        )

    def run(self, params: Mapping[str, float | int | str]) -> SimResult:
        """Evaluate one candidate.

        ``params`` is assumed to have already passed ``validate()`` — the guard
        is the component responsible for rejecting nonsense, not this one.
        """
        started = time.perf_counter()

        # Order matters: the benchmark functions take positional vectors, and
        # the parameter space defines the canonical order.
        values = [float(params[spec.name]) for spec in self._benchmark.space.params]

        objective_value = self._benchmark.objective(values)
        metrics: dict[str, float] = {self._benchmark.objective_metric: objective_value}
        metrics.update(self._benchmark.extra_metrics(values))

        violations = [
            Violation(
                metric=constraint.metric,
                operator=constraint.operator,
                threshold=constraint.threshold,
                actual=metrics[constraint.metric],
                amount=constraint.violation_amount(metrics[constraint.metric]),
            )
            for constraint in self._benchmark.constraints
            if not constraint.is_satisfied(metrics[constraint.metric])
        ]

        return SimResult(
            metrics=metrics,
            objective_value=objective_value,
            constraint_violations=violations,
            feasible=not violations,
            artifacts={},
            wall_time_s=time.perf_counter() - started,
        )
