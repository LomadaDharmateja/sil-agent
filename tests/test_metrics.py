"""Scoring. Every value here is hand-computed, not compared against the code.

These functions decide what the project's headline claim turns out to be, so
they are checked against arithmetic done independently rather than against a
previous run's output.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from sil_agent.agent.state import (
    Best,
    BudgetState,
    Candidate,
    CandidateSource,
    CostRecord,
    Episode,
    Evaluation,
    ReplanDecision,
    RunState,
    RunStatus,
    SimResult,
    TerminationReason,
    ToolError,
    utcnow,
)
from sil_agent.eval.metrics import (
    best_feasible_curve,
    derive_termination,
    evaluations_to_threshold,
    percentile,
    regret_of,
    seed_noise_floor,
    summarise_cell,
    summarise_run,
)
from sil_agent.persistence.repo import StoredRun
from sil_agent.simulators.toy import BRANIN, BRANIN_CONSTRAINED, ToySimulator

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def episode(idx: int, value: float, *, feasible: bool = True) -> Episode:
    return Episode(
        idx=idx,
        candidate=Candidate(params={"x1": 0.0, "x2": 0.0}, source=CandidateSource.BASELINE),
        result=SimResult(
            metrics={"branin": value},
            objective_value=value,
            feasible=feasible,
            wall_time_s=0.001,
        ),
        evaluation=Evaluation.computed_only(
            improved=False, delta_vs_best=0.0, feasible=feasible
        ),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=1,
    )


def rejected_episode(idx: int) -> Episode:
    return Episode(
        idx=idx,
        candidate=Candidate(params={"x1": 0.0, "x2": 0.0}, source=CandidateSource.BASELINE),
        result=ToolError(kind="GuardRejection", message="invented a parameter"),
        evaluation=Evaluation.computed_only(improved=False, delta_vs_best=0.0, feasible=False),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=1,
    )


def stored(
    episodes: list[Episode],
    *,
    status: RunStatus = RunStatus.DONE,
    max_evaluations: int = 10,
    max_rejections: int = 50,
    simulator: str = "branin",
) -> StoredRun:
    goal = ToySimulator.from_name(simulator).default_goal()
    evaluations = sum(1 for e in episodes if e.sim_result is not None)
    rejections = len(episodes) - evaluations
    now = utcnow()

    best: Best | None = None
    for e in episodes:
        result = e.sim_result
        if result is not None and result.better_than(best, goal.objective):
            best = Best(episode_idx=e.idx, candidate=e.candidate, result=result)

    state = RunState(
        run_id=uuid4(),
        goal=goal,
        status=status,
        history=episodes,
        best=best,
        budget=BudgetState(
            max_evaluations=max_evaluations,
            evaluations_used=evaluations,
            max_rejections=max_rejections,
            rejections_used=rejections,
        ),
        step_idx=len(episodes),
        seed=1,
        created_at=now,
        updated_at=now,
    )
    return StoredRun(state=state, simulator=simulator, strategy="random_search", experiment="t")


# ---------------------------------------------------------------------------
# Regret
# ---------------------------------------------------------------------------


def test_regret_is_distance_from_the_known_optimum():
    assert regret_of(1.397887, BRANIN) == pytest.approx(1.0)


def test_regret_is_clamped_at_zero():
    """`known_optimum` is a rounded literal, so a very good run can land below it.

    Reporting a negative regret would say a strategy beat the global optimum.
    """
    assert regret_of(0.3, BRANIN) == 0.0


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def test_curve_tracks_the_running_best():
    curve = best_feasible_curve([episode(0, 5.0), episode(1, 9.0), episode(2, 2.0)], BRANIN)
    assert curve == [5.0, 5.0, 2.0]


def test_curve_ignores_infeasible_results():
    """An infeasible point never becomes the best, however low its objective."""
    curve = best_feasible_curve(
        [episode(0, 5.0), episode(1, -100.0, feasible=False)], BRANIN_CONSTRAINED
    )
    assert curve == [5.0, 5.0]


def test_curve_has_no_entry_for_a_rejected_episode():
    """Rejections consume no evaluation, so they must not shift the x-axis."""
    curve = best_feasible_curve([episode(0, 5.0), rejected_episode(1), episode(2, 3.0)], BRANIN)
    assert curve == [5.0, 3.0]


def test_curve_is_none_until_the_first_feasible_result():
    curve = best_feasible_curve(
        [episode(0, 1.0, feasible=False), episode(1, 8.0)], BRANIN_CONSTRAINED
    )
    assert curve == [None, 8.0]


# ---------------------------------------------------------------------------
# Run summaries
# ---------------------------------------------------------------------------


def test_summary_counts_evaluations_and_rejections_separately():
    summary = summarise_run(stored([episode(0, 5.0), rejected_episode(1), episode(2, 3.0)]), BRANIN)
    assert summary.evaluations == 2
    assert summary.rejections == 1


def test_summary_regret_is_none_when_nothing_was_feasible():
    """Not a large float. A large float would be averaged; None cannot be."""
    summary = summarise_run(
        stored([episode(0, 1.0, feasible=False)], simulator="branin_constrained"),
        BRANIN_CONSTRAINED,
    )
    assert summary.best_objective is None
    assert summary.regret is None


def test_summary_pads_the_curve_to_the_full_budget():
    """A run that stopped early holds its final value.

    That IS its best-so-far at evaluation 10: it never found anything better.
    Without padding, curves of different lengths could not be compared point by
    point across seeds.
    """
    summary = summarise_run(stored([episode(0, 5.0), episode(1, 3.0)], max_evaluations=10), BRANIN)
    assert len(summary.regret_curve) == 10
    assert summary.regret_curve[-1] == summary.regret_curve[1]


def test_evaluations_to_threshold():
    summary = summarise_run(
        stored([episode(0, 50.0), episode(1, 10.0), episode(2, 1.0)], max_evaluations=3), BRANIN
    )
    # regret after each: 49.6, 9.6, 0.602
    assert evaluations_to_threshold(summary, 10.0) == 2
    assert evaluations_to_threshold(summary, 0.001) is None


# ---------------------------------------------------------------------------
# Termination, derived rather than stored
# ---------------------------------------------------------------------------


def test_termination_budget_when_the_allowance_is_spent():
    run = stored([episode(i, 1.0) for i in range(10)], max_evaluations=10)
    assert derive_termination(run) is TerminationReason.BUDGET


def test_termination_exhausted_when_it_stopped_early():
    """Finished, but with evaluations left — the strategy ran out of proposals."""
    run = stored([episode(i, 1.0) for i in range(4)], max_evaluations=10)
    assert derive_termination(run) is TerminationReason.EXHAUSTED


def test_termination_rejections_when_the_guard_kept_refusing():
    run = stored(
        [rejected_episode(i) for i in range(5)],
        max_evaluations=10,
        max_rejections=5,
    )
    assert derive_termination(run) is TerminationReason.REJECTIONS


def test_termination_is_none_for_an_unfinished_run():
    """A run interrupted by Ctrl-C never reaches its final save.

    It must not be mistaken for a strategy that finished early, or the harness
    would skip it instead of resuming it.
    """
    run = stored([episode(0, 1.0)], status=RunStatus.EXECUTING, max_evaluations=10)
    assert derive_termination(run) is None


def test_termination_error_for_a_failed_run():
    run = stored([episode(0, 1.0)], status=RunStatus.FAILED, max_evaluations=10)
    assert derive_termination(run) is TerminationReason.ERROR


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_percentile_interpolates():
    # Between the 2nd (2.0) and 3rd (3.0) of four values: position 0.75*3 = 2.25
    assert percentile([1.0, 2.0, 3.0, 4.0], 75.0) == pytest.approx(3.25)
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)


def test_cell_uses_the_sample_standard_deviation():
    """n-1, not n: five seeds are a sample, not the whole population."""
    summaries = [
        summarise_run(stored([episode(0, value)], max_evaluations=1), BRANIN)
        for value in (1.397887, 2.397887, 3.397887)
    ]
    cell = summarise_cell(summaries)
    assert cell.values == pytest.approx((1.0, 2.0, 3.0))
    assert cell.mean == pytest.approx(2.0)
    assert cell.std == pytest.approx(1.0)  # sample std of 1,2,3 is exactly 1


def test_cell_reports_missing_rather_than_averaging_around_it():
    summaries = [
        summarise_run(stored([episode(0, 1.397887)], max_evaluations=1), BRANIN),
        summarise_run(
            stored(
                [episode(0, 1.0, feasible=False)],
                max_evaluations=1,
                simulator="branin_constrained",
            ),
            BRANIN_CONSTRAINED,
        ),
    ]
    cell = summarise_cell(summaries)
    assert cell.n == 2
    assert cell.missing == 1
    assert cell.mean == pytest.approx(1.0)


def test_a_single_seed_cell_reports_its_value_not_an_absence():
    """A regression test for a report that called every successful run a failure.

    With one seed there is a mean but no *sample* standard deviation, and the
    label used to treat a missing std as missing data — so a 15-evaluation run
    that found the global optimum was printed as "no feasible solution".
    """
    summaries = [summarise_run(stored([episode(0, 1.397887)], max_evaluations=1), BRANIN)]
    cell = summarise_cell(summaries)

    assert cell.mean == pytest.approx(1.0)
    assert cell.std is None, "a sample std needs at least two observations"
    assert cell.label == "1 (n=1)"


def test_cell_label_is_honest_when_nothing_was_feasible():
    summaries = [
        summarise_run(
            stored(
                [episode(0, 1.0, feasible=False)],
                max_evaluations=1,
                simulator="branin_constrained",
            ),
            BRANIN_CONSTRAINED,
        )
    ]
    assert summarise_cell(summaries).label == "no feasible solution"


# ---------------------------------------------------------------------------
# Seed noise floor
# ---------------------------------------------------------------------------


def test_noise_floor_needs_more_seeds_than_the_subsample():
    assert seed_noise_floor([1.0, 2.0], strategy="s", simulator="b", subsample=5) is None


def test_noise_floor_is_deterministic_and_brackets_the_mean():
    values = [float(v) for v in range(30)]
    first = seed_noise_floor(values, strategy="s", simulator="b", subsample=5, iterations=2000)
    second = seed_noise_floor(values, strategy="s", simulator="b", subsample=5, iterations=2000)

    assert first is not None and second is not None
    assert first == second, "the reported interval must be reproducible"
    assert first.low < first.observed_mean < first.high
    assert first.spread > 0
