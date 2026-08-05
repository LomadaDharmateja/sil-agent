"""Improvement comparison — the deterministic half of Rule 2.

``improved`` is computed here and injected into the critic. If this comparison
is wrong then the agent is told it improved when it did not, every phase log
records the wrong "best so far", and the Phase 4 ablation compares noise.
"""

from __future__ import annotations

import pytest

from sil_agent.agent.state import (
    Best,
    Candidate,
    CandidateSource,
    ConstraintOp,
    Direction,
    Objective,
    SimResult,
    Violation,
)

MINIMISE = Objective(metric="f", direction=Direction.MINIMISE)
MAXIMISE = Objective(metric="f", direction=Direction.MAXIMISE)


def result(value: float, *, feasible: bool = True, violation: float = 0.0) -> SimResult:
    violations = []
    if not feasible:
        violations.append(
            Violation(
                metric="g",
                operator=ConstraintOp.LE,
                threshold=0.0,
                actual=violation,
                amount=violation,
            )
        )
    return SimResult(
        metrics={"f": value},
        objective_value=value,
        constraint_violations=violations,
        feasible=feasible,
        wall_time_s=0.0,
    )


def best_of(sim_result: SimResult, idx: int = 0) -> Best:
    return Best(
        episode_idx=idx,
        candidate=Candidate(params={"x": 1.0}, source=CandidateSource.BASELINE),
        result=sim_result,
    )


def test_anything_beats_no_incumbent() -> None:
    assert result(1000.0).better_than(None, MINIMISE)
    assert result(-5.0, feasible=False, violation=3.0).better_than(None, MINIMISE)


def test_lower_is_better_when_minimising() -> None:
    incumbent = best_of(result(10.0))
    assert result(9.0).better_than(incumbent, MINIMISE)
    assert not result(11.0).better_than(incumbent, MINIMISE)


def test_higher_is_better_when_maximising() -> None:
    incumbent = best_of(result(10.0))
    assert result(11.0).better_than(incumbent, MAXIMISE)
    assert not result(9.0).better_than(incumbent, MAXIMISE)


def test_a_tie_is_not_an_improvement() -> None:
    """Strict comparison. A deterministic simulator returning the same value
    twice must not keep resetting the incumbent, or Phase 3's stagnation
    detectors could never fire."""
    incumbent = best_of(result(10.0))
    assert not result(10.0).better_than(incumbent, MINIMISE)
    assert not result(10.0).better_than(incumbent, MAXIMISE)


def test_feasible_beats_infeasible_regardless_of_objective_value() -> None:
    """The rule that makes constrained optimisation behave. An infeasible
    design with a wonderful objective value is not a better design."""
    infeasible_incumbent = best_of(result(-1000.0, feasible=False, violation=5.0))
    assert result(1000.0).better_than(infeasible_incumbent, MINIMISE)


def test_infeasible_never_beats_feasible() -> None:
    feasible_incumbent = best_of(result(1000.0))
    assert not result(-1000.0, feasible=False, violation=5.0).better_than(
        feasible_incumbent, MINIMISE
    )


def test_less_infeasible_beats_more_infeasible() -> None:
    """So that a run starting in an infeasible region can still make progress."""
    incumbent = best_of(result(0.0, feasible=False, violation=10.0))
    assert result(0.0, feasible=False, violation=4.0).better_than(incumbent, MINIMISE)
    assert not result(0.0, feasible=False, violation=20.0).better_than(incumbent, MINIMISE)


def test_delta_is_positive_when_better() -> None:
    incumbent = best_of(result(10.0))
    assert result(7.0).delta_vs(incumbent, MINIMISE) == pytest.approx(3.0)
    assert result(13.0).delta_vs(incumbent, MINIMISE) == pytest.approx(-3.0)
    assert result(13.0).delta_vs(incumbent, MAXIMISE) == pytest.approx(3.0)


def test_delta_against_no_incumbent_is_zero() -> None:
    assert result(10.0).delta_vs(None, MINIMISE) == 0.0


def test_total_violation_sums_the_amounts() -> None:
    sim_result = result(0.0, feasible=False, violation=2.5)
    assert sim_result.total_violation == pytest.approx(2.5)
    assert result(0.0).total_violation == 0.0


def test_models_are_immutable() -> None:
    """Rule 1 enforced by the type system: state cannot be changed in place."""
    sim_result = result(1.0)
    with pytest.raises(ValueError, match="frozen"):
        sim_result.objective_value = 2.0  # type: ignore[misc]


def test_unexpected_fields_are_rejected() -> None:
    """From Phase 3 this is what catches an LLM inventing a field."""
    with pytest.raises(ValueError, match="Extra inputs"):
        Candidate(params={"x": 1.0}, source=CandidateSource.PLANNER, confidence=0.9)  # type: ignore[call-arg]
