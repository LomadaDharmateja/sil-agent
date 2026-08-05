"""Stagnation detection, run locking, and goal parsing.

All three are Phase 3 deliverables that need no provider: the detectors are
arithmetic over history, the lock is tested against a fake Redis, and goal
parsing is tested against a scripted router.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from sil_agent.agent.goal_parser import (
    GoalParseError,
    ParsedConstraint,
    ParsedGoal,
    ParsedObjective,
    validate_goal,
)
from sil_agent.agent.stagnation import (
    DiversityCollapse,
    NoImprovement,
    check_all,
    default_detectors,
    max_pairwise_distance,
    normalise,
)
from sil_agent.agent.state import (
    Candidate,
    CandidateSource,
    ConstraintOp,
    CostRecord,
    Direction,
    Episode,
    Evaluation,
    ReplanDecision,
    SimResult,
    ToolError,
)
from sil_agent.services.locks import (
    LockUnavailable,
    NullLock,
    RunLock,
    lock_key,
    make_token,
)
from sil_agent.simulators.toy import ToySimulator

GOAL = ToySimulator.from_name("branin").default_goal()


def evaluated(idx: int, x1: float, x2: float, *, improved: bool) -> Episode:
    return Episode(
        idx=idx,
        candidate=Candidate(params={"x1": x1, "x2": x2}, source=CandidateSource.PLANNER),
        result=SimResult(
            metrics={"branin": 1.0}, objective_value=1.0, feasible=True, wall_time_s=0.0
        ),
        evaluation=Evaluation.computed_only(
            improved=improved, delta_vs_best=0.0, feasible=True
        ),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=1,
    )


def rejected(idx: int) -> Episode:
    return Episode(
        idx=idx,
        candidate=Candidate(params={"bogus": 1.0}, source=CandidateSource.PLANNER),
        result=ToolError(kind="GuardRejection", message="unknown parameter(s): bogus"),
        evaluation=Evaluation.computed_only(improved=False, delta_vs_best=0.0, feasible=False),
        decision=ReplanDecision.placeholder(),
        cost=CostRecord.zero(),
        duration_ms=1,
    )


# ---------------------------------------------------------------------------
# No improvement
# ---------------------------------------------------------------------------


def test_no_improvement_fires_after_a_flat_window():
    history = [evaluated(i, float(i), 1.0, improved=False) for i in range(8)]
    verdict = NoImprovement(window=8).check(history, GOAL)

    assert verdict.stuck
    assert verdict.detector == "no_improvement"
    assert "8 evaluations" in verdict.detail


def test_no_improvement_stays_quiet_while_progress_continues():
    history = [evaluated(i, float(i), 1.0, improved=i == 5) for i in range(8)]
    assert not NoImprovement(window=8).check(history, GOAL).stuck


def test_no_improvement_needs_a_full_window_first():
    """Three flat evaluations at the start of a run is not a stall."""
    history = [evaluated(i, float(i), 1.0, improved=False) for i in range(3)]
    assert not NoImprovement(window=8).check(history, GOAL).stuck


def test_rejections_do_not_count_towards_the_window():
    """A run whose proposals were all rejected has not had chances to improve.

    Counting them would blame the search for a proposal-quality problem that the
    separate rejection allowance already handles.
    """
    history = [rejected(i) for i in range(20)]
    assert not NoImprovement(window=8).check(history, GOAL).stuck


# ---------------------------------------------------------------------------
# Diversity collapse
# ---------------------------------------------------------------------------


def test_diversity_collapse_fires_when_proposals_crowd_one_point():
    history = [evaluated(i, 3.0 + i * 0.001, 2.0, improved=True) for i in range(6)]
    verdict = DiversityCollapse(window=6, epsilon=0.02).check(history, GOAL)

    assert verdict.stuck, "improving slightly while circling one point is still a stall"
    assert verdict.detector == "diversity_collapse"


def test_diversity_collapse_stays_quiet_while_exploring():
    history = [evaluated(i, -5.0 + i * 3.0, float(i), improved=False) for i in range(6)]
    assert not DiversityCollapse(window=6, epsilon=0.02).check(history, GOAL).stuck


def test_normalisation_scales_each_parameter_by_its_own_bounds():
    """Raw distances are meaningless when x1 spans 15 units and x2 spans 15 from 0.

    Without this, whichever parameter has the widest range dominates the
    detector.
    """
    # branin: x1 in [-5, 10], x2 in [0, 15]
    assert normalise({"x1": -5.0, "x2": 0.0}, GOAL) == pytest.approx([0.0, 0.0])
    assert normalise({"x1": 10.0, "x2": 15.0}, GOAL) == pytest.approx([1.0, 1.0])
    assert normalise({"x1": 2.5, "x2": 7.5}, GOAL) == pytest.approx([0.5, 0.5])


def test_distance_uses_the_diameter_not_the_mean():
    """Five identical points and one far away is still exploration.

    A mean distance would hide the outlier behind the duplicates.
    """
    points = [[0.0, 0.0]] * 5 + [[1.0, 0.0]]
    assert max_pairwise_distance(points) == pytest.approx(1.0)


def test_check_all_returns_the_first_detector_to_fire():
    history = [evaluated(i, 3.0, 2.0, improved=False) for i in range(10)]
    verdict = check_all(default_detectors(), history, GOAL)

    assert verdict.stuck
    assert verdict.detector == "no_improvement", "detectors are checked in order"


def test_no_detectors_means_no_stagnation():
    history = [evaluated(i, 3.0, 2.0, improved=False) for i in range(50)]
    assert not check_all([], history, GOAL).stuck


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


class FakeRedis:
    """Enough Redis to test the lock: SET NX EX, and the two Lua scripts."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def set(self, name, value, *, nx=False, ex=None):
        if nx and name in self.store:
            return None
        self.store[name] = value
        return True

    def eval(self, script, numkeys, *args):
        key, token = args[0], args[1]
        if self.store.get(key) != token:
            return 0
        if "del" in script:
            del self.store[key]
        return 1


def test_a_second_worker_is_refused():
    client = FakeRedis()
    run_id = uuid4()

    first = RunLock(client, run_id)
    first.acquire()

    with pytest.raises(LockUnavailable, match=str(run_id)):
        RunLock(client, run_id).acquire()


def test_releasing_lets_the_next_worker_in():
    client = FakeRedis()
    run_id = uuid4()

    with RunLock(client, run_id):
        pass

    RunLock(client, run_id).acquire()  # must not raise


def test_a_stale_holder_cannot_release_someone_elses_lock():
    """The reason the token exists.

    A worker that stalled past the TTL must not delete the lock a different
    worker has since acquired — both would then be writing to the same run.
    """
    client = FakeRedis()
    run_id = uuid4()

    stale = RunLock(client, run_id, token="stale-worker")
    stale.acquire()
    client.store[lock_key(run_id)] = "new-worker"  # the lease expired and was retaken

    stale.release()

    assert client.store[lock_key(run_id)] == "new-worker", "the new owner still holds it"


def test_renew_reports_when_the_lock_has_been_lost():
    """A False here means stop writing immediately, not 'try again'."""
    client = FakeRedis()
    run_id = uuid4()

    lock = RunLock(client, run_id)
    lock.acquire()
    assert lock.renew() is True

    client.store[lock_key(run_id)] = "someone-else"
    assert lock.renew() is False
    assert lock.held is False


def test_the_null_lock_never_blocks():
    """Running one baseline on a laptop must not require Redis."""
    with NullLock(uuid4()) as lock:
        assert lock.renew() is True


def test_tokens_are_unique_and_traceable():
    first, second = make_token(), make_token()
    assert first != second
    assert ":" in first, "host:pid:random, so a stuck lock can be traced to a process"


# ---------------------------------------------------------------------------
# Goal parsing
# ---------------------------------------------------------------------------

METRICS = ["branin", "x1_plus_x2"]


def test_a_valid_interpretation_becomes_a_goal():
    parsed = ParsedGoal(
        objective=ParsedObjective(metric="branin", direction=Direction.MINIMISE, target=0.5),
        constraints=[
            ParsedConstraint(metric="x1_plus_x2", operator=ConstraintOp.LE, threshold=10.0)
        ],
    )

    goal = validate_goal(
        parsed, raw_text="minimise branin", space=GOAL.parameter_space, reported_metrics=METRICS
    )

    assert goal.objective.metric == "branin"
    assert goal.objective.target == 0.5
    assert len(goal.constraints) == 1
    assert goal.parameter_space is GOAL.parameter_space, "the space comes from the simulator"


def test_a_hallucinated_objective_metric_is_rejected():
    """Fails at episode 0 rather than after fifty episodes of optimising nothing."""
    parsed = ParsedGoal(
        objective=ParsedObjective(metric="fuel_consumption", direction=Direction.MINIMISE)
    )

    with pytest.raises(GoalParseError, match="fuel_consumption"):
        validate_goal(
            parsed, raw_text="x", space=GOAL.parameter_space, reported_metrics=METRICS
        )


def test_a_hallucinated_constraint_metric_is_rejected():
    parsed = ParsedGoal(
        objective=ParsedObjective(metric="branin", direction=Direction.MINIMISE),
        constraints=[ParsedConstraint(metric="mass", operator=ConstraintOp.LE, threshold=1.0)],
    )

    with pytest.raises(GoalParseError, match="mass"):
        validate_goal(
            parsed, raw_text="x", space=GOAL.parameter_space, reported_metrics=METRICS
        )


def test_a_parameter_name_is_not_a_valid_metric():
    """`x1` is an input, not something the simulator reports."""
    parsed = ParsedGoal(objective=ParsedObjective(metric="x1", direction=Direction.MINIMISE))

    with pytest.raises(GoalParseError, match="x1"):
        validate_goal(
            parsed, raw_text="x", space=GOAL.parameter_space, reported_metrics=METRICS
        )
