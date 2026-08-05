"""Grid search and Optuna TPE.

The test that matters most in this file is ``test_tpe_resume_equivalence``.
Both new strategies have a natural implementation that keeps state on the
object — a grid counter, an Optuna ``Study`` — and both of those implementations
pass every test that does not involve restarting. That is the failure this file
exists to catch.
"""

from __future__ import annotations

import random

import pytest

from sil_agent.agent.loop import episode_rng
from sil_agent.agent.state import (
    Candidate,
    CandidateSource,
    CostRecord,
    Episode,
    Evaluation,
    Goal,
    ParameterSpace,
    ParamKind,
    ParamSpec,
    ReplanDecision,
    SimResult,
)
from sil_agent.simulators.toy import ToySimulator
from sil_agent.strategies.base import StrategyExhausted
from sil_agent.strategies.grid import GridSearch, axis_values, points_per_axis
from sil_agent.strategies.optuna_tpe import OptunaTPE, constraint_values

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_episode(idx: int, params: dict[str, float | int | str], value: float) -> Episode:
    """A completed episode with a given objective value."""
    return Episode(
        idx=idx,
        candidate=Candidate(params=params, source=CandidateSource.BASELINE),
        result=SimResult(
            metrics={"f": value},
            objective_value=value,
            feasible=True,
            wall_time_s=0.0,
        ),
        evaluation=Evaluation.computed_only(improved=False, delta_vs_best=0.0, feasible=True),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=0,
    )


def drive(strategy, goal: Goal, simulator: ToySimulator, count: int, seed: int) -> list[Episode]:
    """Run a strategy for `count` episodes without a database, as the loop would."""
    history: list[Episode] = []
    for idx in range(count):
        candidate = strategy.propose(goal, history, episode_rng(seed, idx))
        result = simulator.run(candidate.params)
        history.append(
            Episode(
                idx=idx,
                candidate=candidate,
                result=result,
                evaluation=Evaluation.computed_only(
                    improved=False, delta_vs_best=0.0, feasible=result.feasible
                ),
                decision=ReplanDecision.placeholder(),
                cost=CostRecord.zero(),
                duration_ms=0,
            )
        )
    return history


# ---------------------------------------------------------------------------
# Grid search — sizing
# ---------------------------------------------------------------------------


def test_grid_fits_the_budget_in_two_dimensions():
    space = ToySimulator.from_name("branin").describe()
    assert points_per_axis(space, 200) == 14  # 14^2 = 196 <= 200


def test_grid_collapses_to_two_points_in_six_dimensions():
    """The curse of dimensionality, as a number.

    A 200-evaluation budget buys two points per axis in 6-D. This is the fact
    the comparison table has to report honestly rather than hide.
    """
    space = ToySimulator.from_name("hartmann6").describe()
    assert points_per_axis(space, 200) == 2
    assert GridSearch(200).total_points(space) == 64


def test_grid_sizing_survives_floating_point_truncation():
    """196 ** 0.5 can evaluate to 13.999999999999998.

    Truncating that gives 13 and silently wastes 27 evaluations, so the sizing
    steps up while the larger grid still fits.
    """
    space = ToySimulator.from_name("branin").describe()
    assert points_per_axis(space, 196) == 14


# ---------------------------------------------------------------------------
# Grid search — enumeration
# ---------------------------------------------------------------------------


def test_axis_values_include_both_bounds():
    spec = ParamSpec(name="x", kind=ParamKind.FLOAT, bounds=(-5.0, 10.0))
    values = axis_values(spec, 4)
    assert values[0] == -5.0
    assert values[-1] == 10.0
    assert len(values) == 4


def test_integer_axis_deduplicates():
    """Ten points on an axis spanning [0, 3] is four distinct integers, not ten."""
    spec = ParamSpec(name="n", kind=ParamKind.INT, bounds=(0.0, 3.0))
    assert axis_values(spec, 10) == [0, 1, 2, 3]


def test_categorical_axis_takes_every_choice():
    spec = ParamSpec(name="mode", kind=ParamKind.CATEGORICAL, choices=["a", "b", "c"])
    assert axis_values(spec, 2) == ["a", "b", "c"]


def test_grid_covers_every_point_exactly_once():
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()
    strategy = GridSearch(16)  # 4 x 4

    history = drive(strategy, goal, simulator, 16, seed=1)
    points = [tuple(e.candidate.params.values()) for e in history]

    assert len(points) == 16
    assert len(set(points)) == 16, "grid search proposed the same point twice"


def test_grid_position_comes_from_history_not_a_counter():
    """A fresh strategy handed a 7-episode history must propose the 8th point.

    This is the resume case in miniature. A strategy holding its own counter
    would answer with point 1 and re-evaluate everything already done.
    """
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()
    space = goal.parameter_space

    strategy = GridSearch(16)
    history = [make_episode(idx, strategy.point_at(space, idx), 0.0) for idx in range(7)]

    fresh = GridSearch(16)
    proposed = fresh.propose(goal, history, random.Random(0))

    assert proposed.params == fresh.point_at(space, 7)


def test_grid_raises_when_exhausted():
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()
    strategy = GridSearch(9)  # 3 x 3

    history = [make_episode(idx, {"x1": 0.0, "x2": 0.0}, 0.0) for idx in range(9)]

    with pytest.raises(StrategyExhausted, match="fully covered"):
        strategy.propose(goal, history, random.Random(0))


def test_grid_ignores_the_seed():
    """Grid search is deterministic, so every seed produces the identical run.

    Worth asserting because the comparison table reports a standard deviation of
    exactly zero for it, which otherwise looks like a broken measurement.
    """
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()

    first = drive(GridSearch(16), goal, simulator, 16, seed=1)
    second = drive(GridSearch(16), goal, simulator, 16, seed=99)

    assert [e.candidate.params for e in first] == [e.candidate.params for e in second]


# ---------------------------------------------------------------------------
# Optuna TPE
# ---------------------------------------------------------------------------


def test_tpe_is_deterministic_for_a_seed():
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()

    first = drive(OptunaTPE(), goal, simulator, 20, seed=7)
    second = drive(OptunaTPE(), goal, simulator, 20, seed=7)

    assert [e.candidate.params for e in first] == [e.candidate.params for e in second]


def test_tpe_differs_across_seeds():
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()

    first = drive(OptunaTPE(), goal, simulator, 20, seed=7)
    second = drive(OptunaTPE(), goal, simulator, 20, seed=8)

    assert [e.candidate.params for e in first] != [e.candidate.params for e in second]


def test_tpe_resume_equivalence():
    """The test this phase rests on.

    Run 30 episodes straight through. Then take the first 15, hand them to a
    brand-new strategy object as history, and continue. The two sequences must
    be identical — that is what makes a resumed run the same run.

    An implementation that kept a Study on the object fails here and nowhere
    else.
    """
    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()

    uninterrupted = drive(OptunaTPE(), goal, simulator, 30, seed=7)

    cut = 15
    resumed = list(uninterrupted[:cut])
    fresh = OptunaTPE()  # a new process would have exactly this: no memory
    for idx in range(cut, 30):
        candidate = fresh.propose(goal, resumed, episode_rng(7, idx))
        result = simulator.run(candidate.params)
        resumed.append(
            Episode(
                idx=idx,
                candidate=candidate,
                result=result,
                evaluation=Evaluation.computed_only(
                    improved=False, delta_vs_best=0.0, feasible=result.feasible
                ),
                decision=ReplanDecision.placeholder(),
                cost=CostRecord.zero(),
                duration_ms=0,
            )
        )

    assert [e.candidate.params for e in resumed] == [e.candidate.params for e in uninterrupted]


def test_tpe_learns_something():
    """Sanity check: TPE should beat uniform random on Branin over 60 evaluations.

    Loose on purpose — this guards against the study being rebuilt empty (in
    which case TPE degenerates to random sampling and this fails), not against
    a small change in sampler behaviour.
    """
    from sil_agent.strategies.random_search import RandomSearch

    simulator = ToySimulator.from_name("branin")
    goal = simulator.default_goal()

    tpe = min(e.sim_result.objective_value for e in drive(OptunaTPE(), goal, simulator, 60, seed=3))
    rnd = min(
        e.sim_result.objective_value for e in drive(RandomSearch(), goal, simulator, 60, seed=3)
    )

    assert tpe < rnd


def test_tpe_respects_a_declared_integer_space():
    space = ParameterSpace(
        params=[
            ParamSpec(name="n", kind=ParamKind.INT, bounds=(0.0, 10.0)),
            ParamSpec(name="x", kind=ParamKind.FLOAT, bounds=(0.0, 1.0)),
        ]
    )
    goal = Goal(
        raw_text="test",
        objective=ToySimulator.from_name("branin").default_goal().objective,
        parameter_space=space,
    )

    candidate = OptunaTPE().propose(goal, [], random.Random(1))

    assert isinstance(candidate.params["n"], int)
    assert 0 <= candidate.params["n"] <= 10
    assert isinstance(candidate.params["x"], float)


# ---------------------------------------------------------------------------
# Constraint translation
# ---------------------------------------------------------------------------


def test_constraint_values_report_the_margin_not_just_the_violation():
    """A satisfied constraint has no Violation row, but still has a margin.

    Reading only ``constraint_violations`` would tell Optuna that a point is
    feasible without telling it how close to the edge it sits — which is most of
    what makes constraint-aware sampling work.
    """
    benchmark = ToySimulator.from_name("branin_constrained").benchmark
    constraints = list(benchmark.constraints)

    satisfied = SimResult(
        metrics={"branin": 1.0, "x1_plus_x2": 4.0},
        objective_value=1.0,
        feasible=True,
        wall_time_s=0.0,
    )
    assert constraint_values(satisfied, constraints) == [-6.0]  # 4 - 10

    violated = SimResult(
        metrics={"branin": 1.0, "x1_plus_x2": 13.0},
        objective_value=1.0,
        feasible=False,
        wall_time_s=0.0,
    )
    assert constraint_values(violated, constraints) == [3.0]  # 13 - 10


def test_tpe_finds_feasible_points_under_a_constraint():
    simulator = ToySimulator.from_name("branin_constrained")
    goal = simulator.default_goal()

    history = drive(OptunaTPE(), goal, simulator, 40, seed=5)
    feasible = [e for e in history if e.sim_result is not None and e.sim_result.feasible]

    assert len(feasible) > 20, "constraint-aware TPE should spend most of its evaluations feasible"
