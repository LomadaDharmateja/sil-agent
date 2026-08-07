"""The budget sweep, and the crossover it reports.

The crossover is an *analytical claim* — "TPE overtakes the agent at about N
evaluations" — derived from four measured points. It is the kind of number a
reader will quote, so the interpolation behind it is tested against cases whose
answer is known by construction rather than against a previous run's output.
"""

from __future__ import annotations

import math

import pytest

from sil_agent.eval.metrics import CellStats
from sil_agent.eval.sweep import (
    SweepPoint,
    curve_for,
    expected_n,
    find_crossover,
    significance_row,
    sweep_table,
    uneven_cells,
)


def cell(strategy: str, simulator: str, values: tuple[float, ...]) -> CellStats:
    """A CellStats with only the fields the sweep reads filled in meaningfully."""
    median = sorted(values)[len(values) // 2] if values else None
    return CellStats(
        strategy=strategy,
        simulator=simulator,
        n=len(values),
        values=values,
        missing=0 if values else 3,
        mean=(sum(values) / len(values)) if values else None,
        std=None,
        median=median,
        iqr=None,
        mean_evaluations=0.0,
        terminations=(),
    )


def point(strategy: str, budget: int, regret: float, simulator: str = "branin_i1") -> SweepPoint:
    """A cell whose median is exactly `regret`."""
    return SweepPoint(
        strategy=strategy,
        simulator=simulator,
        budget=budget,
        stats=cell(strategy, simulator, (regret, regret, regret)),
    )


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def test_a_curve_is_ordered_by_budget_whatever_order_the_points_arrive_in():
    points = [point("a", 80, 0.1), point("a", 10, 5.0), point("a", 40, 0.5)]
    assert curve_for(points, "a", "branin_i1") == [(10, 5.0), (40, 0.5), (80, 0.1)]


def test_a_curve_ignores_other_strategies_and_other_benchmarks():
    points = [
        point("a", 10, 1.0),
        point("b", 10, 2.0),
        point("a", 10, 3.0, simulator="hartmann6_i1"),
    ]
    assert curve_for(points, "a", "branin_i1") == [(10, 1.0)]


def test_a_cell_with_no_feasible_result_is_left_out_rather_than_plotted_as_zero():
    empty = SweepPoint(
        strategy="a",
        simulator="branin_i1",
        budget=20,
        stats=cell("a", "branin_i1", ()),
    )
    assert curve_for([point("a", 10, 1.0), empty], "a", "branin_i1") == [(10, 1.0)]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def test_no_crossover_when_one_strategy_leads_throughout():
    points = [
        point("agent", 10, 1.0), point("agent", 100, 0.5),
        point("tpe", 10, 5.0), point("tpe", 100, 2.0),
    ]
    assert find_crossover(points, "agent", "tpe", "branin_i1") is None


def test_crossover_is_found_and_bracketed_by_measured_points():
    # agent leads at 10, TPE leads at 100 — the crossing is between them.
    points = [
        point("agent", 10, 1.0), point("agent", 100, 0.9),
        point("tpe", 10, 4.0), point("tpe", 100, 0.1),
    ]
    crossing = find_crossover(points, "agent", "tpe", "branin_i1")

    assert crossing is not None
    assert crossing.winner_below == "agent"
    assert crossing.winner_above == "tpe"
    assert crossing.lower_budget == 10
    assert crossing.upper_budget == 100
    assert 10 < crossing.estimate < 100


def test_the_crossing_is_interpolated_in_log_space():
    """Both axes span orders of magnitude, so the midpoint is geometric.

    Constructed so the log-regret gap is exactly symmetric about the midpoint:
    at 10 the agent is 10x better, at 1000 it is 10x worse. The crossing
    therefore sits at the geometric centre, 100 — not the arithmetic one, 505.
    """
    points = [
        point("agent", 10, 1.0), point("agent", 1000, 10.0),
        point("tpe", 10, 10.0), point("tpe", 1000, 1.0),
    ]
    crossing = find_crossover(points, "agent", "tpe", "branin_i1")

    assert crossing is not None
    assert crossing.estimate == pytest.approx(100.0, rel=1e-6)


def test_the_crossing_lands_on_a_measured_point_when_the_curves_meet_there():
    points = [
        point("agent", 10, 1.0), point("agent", 40, 2.0), point("agent", 80, 4.0),
        point("tpe", 10, 4.0), point("tpe", 40, 2.0), point("tpe", 80, 1.0),
    ]
    crossing = find_crossover(points, "agent", "tpe", "branin_i1")

    assert crossing is not None
    assert crossing.estimate == pytest.approx(40.0, rel=1e-6)


def test_only_the_first_crossing_is_reported():
    """Curves that weave are reported at their first swap, not their last.

    With four budgets a second crossing is far more likely to be noise than a
    real reversal, and reporting the last one would silently prefer the noisier
    end of the sweep.
    """
    points = [
        point("agent", 10, 1.0), point("agent", 40, 5.0), point("agent", 80, 1.0),
        point("tpe", 10, 5.0), point("tpe", 40, 1.0), point("tpe", 80, 5.0),
    ]
    crossing = find_crossover(points, "agent", "tpe", "branin_i1")

    assert crossing is not None
    assert (crossing.lower_budget, crossing.upper_budget) == (10, 40)


def test_a_crossover_needs_two_shared_budgets():
    """One strategy measured at a budget the other was not tells us nothing."""
    points = [point("agent", 10, 1.0), point("tpe", 40, 2.0)]
    assert find_crossover(points, "agent", "tpe", "branin_i1") is None


def test_describe_names_both_sides_and_the_bracket():
    points = [
        point("agent", 20, 0.4), point("agent", 80, 0.4),
        point("tpe", 20, 2.0), point("tpe", 80, 0.1),
    ]
    crossing = find_crossover(points, "agent", "tpe", "branin_i1")

    assert crossing is not None
    text = crossing.describe()
    assert "agent" in text and "tpe" in text
    assert "20" in text and "80" in text


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def test_the_table_has_a_column_per_budget_and_a_row_per_strategy():
    points = [
        point("agent", 10, 1.0), point("agent", 40, 0.5),
        point("tpe", 10, 4.0), point("tpe", 40, 0.9),
    ]
    table = sweep_table(points, "branin_i1")

    assert "10 evals" in table and "40 evals" in table
    assert "| agent |" in table and "| tpe |" in table


def test_a_missing_cell_is_a_dash_rather_than_a_silently_shifted_column():
    """A gap in one row must not slide the remaining values into wrong columns."""
    points = [
        point("agent", 10, 1.0), point("agent", 80, 0.3),
        point("tpe", 10, 4.0), point("tpe", 40, 0.9), point("tpe", 80, 0.2),
    ]
    table = sweep_table(points, "branin_i1")

    agent_row = next(line for line in table.splitlines() if line.startswith("| agent |"))
    # name + three budgets = five pipes
    assert agent_row.count("|") == 5
    assert "—" in agent_row


def test_a_strategy_measured_at_one_budget_is_left_out_of_the_sweep():
    """The sweep is about slopes, and a single point has none.

    `phase35-main` carries all five strategies but only the agent and TPE were
    re-run at other budgets. Including the rest produced a row of dashes that
    reads as missing data rather than as "not part of this experiment".
    """
    points = [
        point("agent", 10, 1.0), point("agent", 40, 0.5),
        point("tpe", 10, 4.0), point("tpe", 40, 0.9),
        point("grid_search", 20, 2.2),
    ]
    table = sweep_table(points, "branin_i1")

    assert "grid_search" not in table
    assert "| agent |" in table and "| tpe |" in table


def test_a_budget_only_a_dropped_strategy_had_does_not_become_an_empty_column():
    points = [
        point("agent", 10, 1.0), point("agent", 40, 0.5),
        point("tpe", 10, 4.0), point("tpe", 40, 0.9),
        point("grid_search", 20, 2.2),
    ]
    table = sweep_table(points, "branin_i1")

    assert "20 evals" not in table


# ---------------------------------------------------------------------------
# Per-budget significance
# ---------------------------------------------------------------------------


def spread(strategy: str, budget: int, values: tuple[float, ...]) -> SweepPoint:
    return SweepPoint(strategy, "branin_i1", budget, cell(strategy, "branin_i1", values))


def test_separated_distributions_are_significant_and_overlapping_ones_are_not():
    """The check that stops a median crossing being read as a result.

    At budget 10 the two groups are cleanly separated; at budget 40 they
    interleave. The medians differ in both cases — only one of them means
    anything.
    """
    points = [
        spread("agent", 10, (0.1, 0.2, 0.3, 0.4, 0.5)),
        spread("tpe", 10, (1.1, 1.2, 1.3, 1.4, 1.5)),
        spread("agent", 40, (0.1, 0.9, 1.1, 1.9, 2.1)),
        spread("tpe", 40, (0.2, 0.8, 1.2, 1.8, 2.2)),
    ]

    results = {budget: (p, leader) for budget, _, p, leader in
               significance_row(points, "branin_i1", "agent", "tpe")}

    assert results[10][0] < 0.05
    assert results[10][1] == "agent"
    assert results[40][0] > 0.5


def test_a_budget_measured_for_only_one_strategy_is_skipped():
    points = [
        spread("agent", 10, (0.1, 0.2, 0.3)),
        spread("tpe", 10, (1.1, 1.2, 1.3)),
        spread("tpe", 80, (0.1, 0.2, 0.3)),
    ]
    budgets = [b for b, _, _, _ in significance_row(points, "branin_i1", "agent", "tpe")]
    assert budgets == [10]


def test_a_single_seed_cell_is_skipped_rather_than_tested():
    """A rank test on one observation reports a number that means nothing."""
    points = [spread("agent", 10, (0.1,)), spread("tpe", 10, (1.1, 1.2, 1.3))]
    assert significance_row(points, "branin_i1", "agent", "tpe") == []


# ---------------------------------------------------------------------------
# Unequal seeds
# ---------------------------------------------------------------------------


def test_a_cell_with_fewer_seeds_than_its_neighbours_is_flagged():
    """An interrupted budget must not pass as a full one.

    A median over two runs prints exactly like a median over five and draws the
    same marker. The only thing that distinguishes them is being told.
    """
    points = [
        SweepPoint("agent", "branin_i1", 10, cell("agent", "branin_i1", (1.0, 1.1, 1.2, 1.3, 1.4))),
        SweepPoint("agent", "branin_i1", 80, cell("agent", "branin_i1", (0.3, 0.4))),
        SweepPoint("tpe", "branin_i1", 10, cell("tpe", "branin_i1", (4.0, 4.1, 4.2, 4.3, 4.4))),
        SweepPoint("tpe", "branin_i1", 80, cell("tpe", "branin_i1", (0.2, 0.3, 0.4, 0.5, 0.6))),
    ]

    flagged = uneven_cells(points, "branin_i1")

    assert [(p.strategy, p.budget) for p in flagged] == [("agent", 80)]
    assert expected_n(points, "branin_i1") == 5


def test_an_even_sweep_is_not_flagged():
    points = [
        SweepPoint("agent", "branin_i1", 10, cell("agent", "branin_i1", (1.0, 1.1, 1.2))),
        SweepPoint("agent", "branin_i1", 80, cell("agent", "branin_i1", (0.3, 0.4, 0.5))),
        SweepPoint("tpe", "branin_i1", 10, cell("tpe", "branin_i1", (4.0, 4.1, 4.2))),
        SweepPoint("tpe", "branin_i1", 80, cell("tpe", "branin_i1", (0.2, 0.3, 0.4))),
    ]
    assert uneven_cells(points, "branin_i1") == []


def test_a_cell_with_no_feasible_result_is_not_reported_as_merely_uneven():
    """Zero seeds is a different problem, already handled by dropping the point."""
    points = [
        SweepPoint("agent", "branin_i1", 10, cell("agent", "branin_i1", (1.0, 1.1, 1.2))),
        SweepPoint("agent", "branin_i1", 80, cell("agent", "branin_i1", ())),
        SweepPoint("tpe", "branin_i1", 10, cell("tpe", "branin_i1", (4.0, 4.1, 4.2))),
        SweepPoint("tpe", "branin_i1", 80, cell("tpe", "branin_i1", (0.2, 0.3, 0.4))),
    ]
    assert uneven_cells(points, "branin_i1") == []


def test_log_floor_keeps_a_zero_regret_off_the_end_of_a_log_axis():
    """A strategy that finds the exact optimum must not break the interpolation."""
    points = [
        point("agent", 10, 0.0), point("agent", 80, 0.0),
        point("tpe", 10, 5.0), point("tpe", 80, 1.0),
    ]
    crossing = find_crossover(points, "agent", "tpe", "branin_i1")
    assert crossing is None  # agent leads throughout; no infinities raised

    for _, regret in curve_for(points, "agent", "branin_i1"):
        assert math.isfinite(regret)
