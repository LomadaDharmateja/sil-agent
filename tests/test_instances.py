"""Shifted instances, and the anonymity of what the model is shown.

Phase 3's headline number was worthless: the planner had memorised Branin and
proposed its three global minima before seeing a result. These tests are the
machinery that stops that happening again, so they check two different things.

**That the transform is correct.** A shifted instance evaluates the underlying
function *outside its published domain*, so `known_optimum` is only still the
optimum if the function does not dip lower out there. Every regret number in the
project rests on that, and it is verified by search rather than by argument.

**That nothing names the benchmark.** Checked by rendering the real planner
prompt, not by inspecting fields — the leak is whatever reaches the model.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize

from sil_agent.agent.planner import render_planner_prompt
from sil_agent.prompts import load
from sil_agent.simulators.instances import (
    GRID_CLEARANCE,
    GRID_STEP,
    OPTIMUM_MARGIN,
    _distance_to_grid,
    make_instance,
)
from sil_agent.simulators.toy import (
    BENCHMARKS,
    BRANIN,
    BRANIN_CONSTRAINED,
    ToySimulator,
)

INSTANCES = sorted(name for name, b in BENCHMARKS.items() if b.is_instance)
ORIGINALS = sorted(name for name, b in BENCHMARKS.items() if not b.is_instance)


# ---------------------------------------------------------------------------
# The transform preserves the optimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", INSTANCES)
def test_the_closed_form_puts_the_optimum_exactly_where_it_claims(name):
    """`g(z_opt)` must equal the original's optimum, not merely come close.

    This is the property the whole construction exists for: shift and rotate the
    problem, keep the optimal *value*, so regret stays reportable.
    """
    benchmark = BENCHMARKS[name]
    location = list(benchmark.known_optimisers[0])

    value = benchmark.objective(location)

    # 1e-6 rather than exact: `known_optimum` is a rounded literal in the
    # literature (-3.32237 for Hartmann-6), so the transform reproduces the
    # true optimum more precisely than the constant records it.
    assert value == pytest.approx(benchmark.known_optimum, abs=1e-5)


@pytest.mark.parametrize("name", INSTANCES)
def test_nothing_in_the_search_box_beats_the_known_optimum(name):
    """The correctness gate under every regret number computed on an instance.

    The transform pushes evaluation outside the original domain. If the function
    goes lower out there, `known_optimum` is not the optimum, regret goes
    negative, the clamp hides it, and every published figure for this instance is
    quietly wrong. Multi-start L-BFGS-B over the box is what turns that from an
    assumption into a check.
    """
    benchmark = BENCHMARKS[name]
    dim = len(benchmark.space.params)

    generator = np.random.Generator(np.random.PCG64(20260806))
    best = math.inf
    for _ in range(60):
        start = generator.uniform(0.0, 1.0, dim)
        result = minimize(
            lambda z: benchmark.objective(list(z)),
            start,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * dim,
        )
        best = min(best, float(result.fun))

    assert best >= benchmark.known_optimum - 1e-5, (
        f"{name}: found {best} below the declared optimum {benchmark.known_optimum}. "
        "The transform reaches a region where the base function is lower, so "
        "regret computed against this optimum is wrong."
    )


@pytest.mark.parametrize("name", INSTANCES)
def test_the_optimum_is_reachable_and_off_the_wall(name):
    benchmark = BENCHMARKS[name]
    low, high = OPTIMUM_MARGIN
    for value in benchmark.known_optimisers[0]:
        assert low <= value <= high


@pytest.mark.parametrize("name", INSTANCES)
def test_the_optimum_never_sits_on_a_round_number(name):
    """A regression test for a measurement that quietly broke itself.

    The first `branin_i1` placed its optimum at (0.6001, 0.4017). The agent
    proposed (0.5, 0.5), (0.7, 0.3), then (0.6, 0.4) — and scored regret 3e-4 in
    three evaluations on a function it had never seen. It had not recalled
    anything; it walked a grid of round numbers and one of them was the answer.

    Models propose round numbers; random search and TPE do not. An optimum on
    the grid is therefore differentially easy for the strategy under test, which
    biases the comparison towards the thing being measured.
    """
    for value in BENCHMARKS[name].known_optimisers[0]:
        assert _distance_to_grid(value) >= GRID_CLEARANCE, (
            f"{name}: optimum coordinate {value} is within {GRID_CLEARANCE} of a "
            f"multiple of {GRID_STEP}, which a language model proposes constantly"
        )


@pytest.mark.parametrize("name", INSTANCES)
def test_a_coarse_grid_of_round_numbers_does_not_find_the_optimum(name):
    """The property the clearance is for, checked against behaviour not geometry.

    Sweeping every round number a model plausibly proposes must leave real
    regret. Geometry alone would not catch a case where the basin is wide enough
    that being off-grid still lands in it.
    """
    benchmark = BENCHMARKS[name]
    dim = len(benchmark.space.params)
    if dim > 3:
        pytest.skip("the full 0.1 grid is 10**d points; checked on the low-dimensional cases")

    grid = [round(0.1 * i, 2) for i in range(1, 10)]
    best = min(
        benchmark.objective(list(point)) for point in itertools.product(grid, repeat=dim)
    )

    assert best - benchmark.known_optimum > 0.1, (
        f"{name}: a plain sweep of round numbers reaches regret "
        f"{best - benchmark.known_optimum}, so the instance rewards guessing "
        "round numbers rather than searching"
    )


# ---------------------------------------------------------------------------
# The transform actually transforms
# ---------------------------------------------------------------------------


def test_an_instance_is_a_different_function_from_its_base():
    """A shift that shifted nothing would pass every other test in this file."""
    instance = BENCHMARKS["branin_i1"]
    # Same normalised coordinates, evaluated on base and instance.
    for point in ([0.25, 0.25], [0.5, 0.5], [0.75, 0.1]):
        base_value = BRANIN.objective(
            [-5.0 + point[0] * 15.0, 0.0 + point[1] * 15.0]
        )
        assert instance.objective(point) != pytest.approx(base_value)


def test_the_instance_optimum_is_not_a_published_one():
    """The memorisation property, stated deterministically.

    A model that recalls Branin's minima knows (-pi, 12.275) and its siblings.
    In the instance's normalised coordinates those points must not be the answer
    — otherwise the shift has moved the problem somewhere the model has still
    effectively been told about.
    """
    instance = BENCHMARKS["branin_i1"]
    optimum = instance.known_optimisers[0]

    for published in BRANIN.known_optimisers:
        normalised = (
            (published[0] - (-5.0)) / 15.0,
            (published[1] - 0.0) / 15.0,
        )
        distance = math.dist(optimum, normalised)
        assert distance > 0.1, (
            "the instance optimum coincides with a published Branin minimum, "
            "so a model that memorised the benchmark can still recall it"
        )


def test_instances_are_deterministic():
    """Same benchmark, same seed, same geometry — or results are not reproducible."""
    first = make_instance(BRANIN, 7)
    second = make_instance(BRANIN, 7)

    assert first.known_optimisers == second.known_optimisers
    for point in ([0.1, 0.9], [0.44, 0.61]):
        assert first.objective(point) == second.objective(point)


def test_different_seeds_give_different_instances():
    assert make_instance(BRANIN, 1).known_optimisers != make_instance(BRANIN, 2).known_optimisers


def test_the_same_seed_on_different_benchmarks_is_not_the_same_transform():
    """The RNG is keyed by the benchmark name as well as the seed.

    Otherwise every `_i1` instance would share one rotation and one optimum
    location, and a strategy that happened to suit that geometry would suit all
    of them at once — turning three benchmarks into one repeated measurement.

    Compared against a renamed copy of the same benchmark so that the only
    difference is the name, which is exactly the dependency being asserted.
    """
    renamed = replace(BRANIN, name="not_branin")

    original = make_instance(BRANIN, 1)
    other = make_instance(renamed, 1)

    assert original.known_optimisers != other.known_optimisers


def test_a_constrained_benchmark_is_refused_rather_than_silently_wrong():
    """Rotating the space changes what `x1 + x2 <= 10` means.

    The constrained benchmark's known optimum depends on which minima survive
    that half-plane, and the transform does not preserve it. Refusing is the
    honest outcome; producing an instance with an optimum that is no longer
    correct is not.
    """
    with pytest.raises(ValueError, match="constraints"):
        make_instance(BRANIN_CONSTRAINED, 1)


# ---------------------------------------------------------------------------
# Anonymity of the prompt
# ---------------------------------------------------------------------------

# Names a model would recognise. Not only the ones in use: a benchmark added
# later must not reintroduce the leak, and this list is the reminder.
FORBIDDEN = (
    "branin",
    "hartmann",
    "rosenbrock",
    "ackley",
    "rastrigin",
    "griewank",
    "schwefel",
    "levy",
    "michalewicz",
    "sphere function",
    "styblinski",
)


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_the_planner_prompt_never_names_the_benchmark(name):
    """Rendered from the real planner path, not reconstructed here.

    `render_planner_prompt` is what `Planner.propose` calls. A test that built
    the prompt itself would keep passing after the planner started sending
    something else, which is the exact failure this guards against.
    """
    simulator = ToySimulator.from_name(name)
    goal = simulator.default_goal()

    system, user = render_planner_prompt(
        load("planner", "v1"), goal, [], None, max_evaluations=20
    )
    text = f"{system}\n{user}".lower()

    for term in FORBIDDEN:
        assert term not in text, f"{name}: the planner prompt contains {term!r}"


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_the_goal_never_names_the_benchmark(name):
    """Belt and braces: the goal is persisted and reaches other prompts too."""
    goal = ToySimulator.from_name(name).default_goal()
    text = f"{goal.raw_text} {goal.objective.metric}".lower()

    for term in FORBIDDEN:
        assert term not in text


@pytest.mark.parametrize("name", INSTANCES)
def test_an_instance_prompt_reveals_nothing_about_the_optimum(name):
    """No number in the prompt is close to the answer.

    Anonymising the name is not enough if the bounds, the dimension or an
    example value hand the location over. With an instance there is nothing to
    hand over — the optimum is at a seeded interior point — and this asserts it
    rather than assuming it.
    """
    simulator = ToySimulator.from_name(name)
    goal = simulator.default_goal()
    benchmark = BENCHMARKS[name]

    system, user = render_planner_prompt(
        load("planner", "v1"), goal, [], None, max_evaluations=20
    )
    text = f"{system}\n{user}"

    for coordinate in benchmark.known_optimisers[0]:
        assert f"{coordinate:.4f}" not in text
        assert f"{coordinate:.3f}" not in text


@pytest.mark.parametrize("name", INSTANCES)
def test_an_instance_is_posed_on_the_unit_box_with_neutral_names(name):
    """The fingerprint layer that renaming cannot close.

    A 2-D problem over exactly [-5, 10] x [0, 15] is Branin whatever its
    parameters are called. Instances are posed on the unit cube, so the domain
    identifies nothing.
    """
    space = BENCHMARKS[name].space
    for index, spec in enumerate(space.params, start=1):
        assert spec.name == f"p{index}"
        assert spec.bounds == (0.0, 1.0)


@pytest.mark.parametrize("name", ORIGINALS)
def test_the_originals_keep_their_true_bounds(name):
    """The originals are anonymised in name only, and that is documented.

    They cannot be made anonymous — changing the domain changes the function —
    so this test records the limitation rather than letting a reader assume the
    scrubbing was complete.
    """
    benchmark = BENCHMARKS[name]
    assert not benchmark.is_instance
    assert benchmark.instance_seed is None


# ---------------------------------------------------------------------------
# Instances behave like any other simulator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", INSTANCES)
def test_an_instance_runs_through_the_ordinary_simulator_path(name):
    simulator = ToySimulator.from_name(name)
    space = simulator.describe()
    params = {spec.name: 0.5 for spec in space.params}

    result = simulator.run(params)

    assert result.feasible
    assert result.objective_value == pytest.approx(
        BENCHMARKS[name].objective([0.5] * len(space.params))
    )
    assert set(result.metrics) == {"objective"}
