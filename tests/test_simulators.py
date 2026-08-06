"""Benchmark correctness.

These are the most important tests in the phase. If Branin does not evaluate to
0.397887 at (pi, 2.275), then every number this project ever reports — every
regret figure, every claim that the agent beat TPE — is measured against the
wrong function.

Each test checks the value *and* the location. A function can return a
plausible-looking minimum while being subtly wrong; being right at the
documented optimiser as well is much harder to achieve by accident.
"""

from __future__ import annotations

import math
import random

import pytest

from sil_agent.agent.state import Direction, ParamKind
from sil_agent.simulators.toy import (
    BENCHMARKS,
    ROSENBROCK_DIM,
    ToySimulator,
    branin,
    hartmann6,
    rosenbrock,
)

BRANIN_OPTIMUM = 0.397887
HARTMANN6_OPTIMUM = -3.32237


# ---------------------------------------------------------------------------
# The functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x1", "x2"),
    [
        (-math.pi, 12.275),
        (math.pi, 2.275),
        (3.0 * math.pi, 2.475),
    ],
)
def test_branin_hits_known_optimum_at_all_three_minima(x1: float, x2: float) -> None:
    assert branin(x1, x2) == pytest.approx(BRANIN_OPTIMUM, abs=1e-6)


def test_branin_optimum_is_actually_a_minimum() -> None:
    """Nothing nearby should be better. Catches a sign or coefficient error."""
    rng = random.Random("branin-search")
    for _ in range(5000):
        x1 = rng.uniform(-5.0, 10.0)
        x2 = rng.uniform(0.0, 15.0)
        assert branin(x1, x2) >= BRANIN_OPTIMUM - 1e-6


def test_hartmann6_hits_known_optimum() -> None:
    """The unrescaled form, whose documented minimum is -3.32237.

    The reference MATLAB implementation at sfu.ca returns a rescaled variant,
    -(2.58 + outer) / 1.94, which minimises at a completely different value.
    Getting that one instead is the single easiest way to build this wrong, and
    this assertion is what catches it.

    The tolerance is 1e-4 because the published optimiser is itself rounded to
    six decimal places, not because the function is approximate.
    """
    x_star = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    assert hartmann6(x_star) == pytest.approx(HARTMANN6_OPTIMUM, abs=1e-4)


def test_hartmann6_optimum_is_actually_a_minimum() -> None:
    rng = random.Random("hartmann6-search")
    for _ in range(5000):
        x = [rng.uniform(0.0, 1.0) for _ in range(6)]
        assert hartmann6(x) >= HARTMANN6_OPTIMUM - 1e-4


def test_hartmann6_rejects_wrong_dimensionality() -> None:
    with pytest.raises(ValueError, match="6 values"):
        hartmann6([0.5, 0.5, 0.5])


def test_rosenbrock_is_exactly_zero_at_all_ones() -> None:
    """Not approximately zero — exactly. Every term is (1 - 1) or (1 - 1^2)."""
    assert rosenbrock([1.0] * ROSENBROCK_DIM) == 0.0


def test_rosenbrock_known_values() -> None:
    # f(0, 0) = 100*(0 - 0)^2 + (0 - 1)^2 = 1
    assert rosenbrock([0.0, 0.0]) == pytest.approx(1.0)
    # f(2, 2) = 100*(2 - 4)^2 + (2 - 1)^2 = 401
    assert rosenbrock([2.0, 2.0]) == pytest.approx(401.0)


def test_rosenbrock_is_never_negative() -> None:
    rng = random.Random("rosenbrock-search")
    for _ in range(2000):
        x = [rng.uniform(-5.0, 10.0) for _ in range(ROSENBROCK_DIM)]
        assert rosenbrock(x) >= 0.0


# ---------------------------------------------------------------------------
# The simulator wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_every_benchmark_reports_its_documented_optimum(name: str) -> None:
    """Run each benchmark through the simulator at its documented optimiser."""
    simulator = ToySimulator.from_name(name)
    benchmark = simulator.benchmark

    for optimiser in benchmark.known_optimisers:
        params = {
            spec.name: value for spec, value in zip(benchmark.space.params, optimiser, strict=True)
        }
        result = simulator.run(params)
        assert result.objective_value == pytest.approx(benchmark.known_optimum, abs=1e-4)
        # A documented optimiser of a constrained problem must be feasible.
        assert result.feasible


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_describe_matches_the_functions_domain(name: str) -> None:
    space = ToySimulator.from_name(name).describe()
    assert space.params
    for spec in space.params:
        assert spec.kind is ParamKind.FLOAT
        assert spec.bounds is not None
        low, high = spec.bounds
        assert low < high


def test_unknown_benchmark_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown benchmark"):
        ToySimulator.from_name("not_a_benchmark")


def test_simulator_is_deterministic() -> None:
    simulator = ToySimulator.from_name("hartmann6")
    params: dict[str, float | int | str] = {f"x{i}": 0.3 for i in range(1, 7)}
    first = simulator.run(params)
    second = simulator.run(params)
    assert first.objective_value == second.objective_value


def test_default_goal_uses_the_simulators_own_parameter_space() -> None:
    """The simulator is the authority on what parameters exist."""
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()
    assert goal.parameter_space == simulator.describe()
    assert goal.objective.direction is Direction.MINIMISE
    # `objective`, not `branin`: the metric name is rendered into every planner
    # prompt, so naming the function here told the model which function it was.
    # See tests/test_instances.py for the anonymity checks this supports.
    assert goal.objective.metric == "objective"


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_constrained_branin_flags_a_violation() -> None:
    simulator = ToySimulator.from_name("branin_constrained")
    # x1 + x2 = 16 > 10
    result = simulator.run({"x1": 8.0, "x2": 8.0})

    assert not result.feasible
    assert len(result.constraint_violations) == 1
    violation = result.constraint_violations[0]
    assert violation.metric == "x1_plus_x2"
    assert violation.actual == pytest.approx(16.0)
    assert violation.amount == pytest.approx(6.0)


def test_constrained_branin_accepts_a_feasible_point() -> None:
    simulator = ToySimulator.from_name("branin_constrained")
    # x1 + x2 = 5.42 <= 10
    result = simulator.run({"x1": math.pi, "x2": 2.275})

    assert result.feasible
    assert result.constraint_violations == []
    assert result.metrics["x1_plus_x2"] == pytest.approx(math.pi + 2.275)


def test_constraint_boundary_is_inclusive_for_le() -> None:
    simulator = ToySimulator.from_name("branin_constrained")
    result = simulator.run({"x1": 5.0, "x2": 5.0})  # exactly 10
    assert result.feasible


def test_unconstrained_branin_has_no_constraints() -> None:
    simulator = ToySimulator.from_name("branin")
    result = simulator.run({"x1": 8.0, "x2": 8.0})
    assert result.feasible
    assert result.constraint_violations == []
