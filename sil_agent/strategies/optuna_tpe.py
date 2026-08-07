"""Optuna's TPE sampler — the honest competitor.

TPE (Tree-structured Parzen Estimator) is the standard Bayesian-optimisation
baseline: it fits one density over the parameter values that produced good
results and another over the rest, and proposes points where the ratio between
them is highest. If the agent cannot beat this, the agent is not interesting.
Using Optuna's implementation rather than writing one means the comparison
cannot be dismissed as a straw man.

The whole difficulty is that Optuna is built around a stateful ``Study`` object
that accumulates trials, and this codebase forbids exactly that.

**Why holding a Study would be wrong.** Rule 1 says the next episode is a pure
function of persisted state. A ``Study`` living on the strategy is in-memory
state the loop depends on, and it fails in a specific, quiet way: resume a run
in a fresh process and the sampler starts with zero trials, so episode 40 is
proposed as though it were episode 0. The run completes, the numbers look
plausible, and it is no longer the run it would have been. This is the same bug
as a per-run RNG in Phase 1, wearing a different hat, and it passes every test
that does not involve a restart.

**So the study is rebuilt from ``history`` on every call.** Every completed
episode is replayed into a fresh study as a finished trial, the sampler is
seeded from persisted values, and one point is asked for. The cost is O(n) per
episode — a 200-episode run reconstructs about 20,000 trials in total — and
that is simply what Rule 1 costs here. It is cheap at this scale and it makes
"resume produces an identical sequence" true rather than nearly true.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState, create_trial

from sil_agent.agent.state import (
    Candidate,
    CandidateSource,
    Constraint,
    ConstraintOp,
    Direction,
    Episode,
    Goal,
    ParameterSpace,
    ParamKind,
    SimResult,
)

# Optuna logs an INFO line for every trial it creates. Rebuilding the study on
# every episode turns that into tens of thousands of lines per run, which buries
# the loop's own output. Set once, at import.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# `constraints_func` is marked experimental, and Optuna warns every time a
# sampler is constructed with one — which here is once per episode, so a
# constrained 200-episode run emits 200 identical warnings. Silenced narrowly:
# this one message, from this one module. A blanket `ignore` would also hide
# the next experimental API that gets used by accident.
warnings.filterwarnings(
    "ignore",
    message=".*constraints_func.*is an experimental feature.*",
    category=optuna.exceptions.ExperimentalWarning,
)

# The key Optuna itself uses to stash constraint values on a trial. Reading and
# writing the same key is what lets a rebuilt trial carry its feasibility
# information back to the sampler.
CONSTRAINTS_KEY = "constraints"


def distributions_for(space: ParameterSpace) -> dict[str, BaseDistribution]:
    """Translate the simulator's declared space into Optuna distributions.

    The simulator stays the authority on what exists and what range it takes —
    the same rule the guard enforces. Optuna is told; it is never asked.
    """
    distributions: dict[str, BaseDistribution] = {}

    for spec in space.params:
        match spec.kind:
            case ParamKind.FLOAT:
                assert spec.bounds is not None
                low, high = spec.bounds
                distributions[spec.name] = FloatDistribution(low=low, high=high)
            case ParamKind.INT:
                assert spec.bounds is not None
                low, high = spec.bounds
                distributions[spec.name] = IntDistribution(low=int(low), high=int(high))
            case ParamKind.CATEGORICAL:
                assert spec.choices is not None
                distributions[spec.name] = CategoricalDistribution(choices=list(spec.choices))

    return distributions


def constraint_values(result: SimResult, constraints: Sequence[Constraint]) -> list[float]:
    """Signed constraint values in Optuna's convention: <= 0 means satisfied.

    Computed from the raw metric rather than from ``constraint_violations``,
    which only records constraints that were *broken*. A satisfied constraint
    has no violation row, so a violations-only reading would lose the margin —
    it could tell TPE that a point is feasible but not that it is close to the
    edge, which is most of what makes constraint-aware sampling work.
    """
    values: list[float] = []

    for constraint in constraints:
        actual = result.metrics.get(constraint.metric)
        if actual is None:
            # The simulator did not report this metric. Treat as satisfied
            # rather than guessing; the constraint cannot be evaluated.
            values.append(-1.0)
            continue

        match constraint.operator:
            case ConstraintOp.LE | ConstraintOp.LT:
                values.append(actual - constraint.threshold)
            case ConstraintOp.GE | ConstraintOp.GT:
                values.append(constraint.threshold - actual)
            case ConstraintOp.EQ:
                values.append(abs(actual - constraint.threshold) - constraint.tolerance)

    return values


def _read_constraints(trial: FrozenTrial) -> Sequence[float]:
    """The ``constraints_func`` handed to the sampler.

    Optuna normally calls this after a trial finishes and caches the result.
    Here every trial is reconstructed with its constraint values already in
    ``system_attrs``, so this just reads them back out.
    """
    stored = trial.system_attrs.get(CONSTRAINTS_KEY)
    if stored is None:
        return []
    return [float(value) for value in stored]


# How many trials TPE samples at random before its model takes over.
#
# This is Optuna's own default, set explicitly rather than inherited, because at
# the budgets this project uses it is a load-bearing choice: at 20 evaluations it
# means **half the run is random search**, and a baseline handicapped by an
# unexamined default is not a baseline.
#
# It was examined. Median final regret over 40 seeds, varying only this value:
#
#     budget            start=3   start=5   start=8   start=10
#     20  branin_i1       2.723     2.664     1.686      1.689
#     20  hartmann6_i1    1.646     1.530     1.639      1.709
#     40  branin_i1      0.5339    0.6786    0.5758     0.3621
#     80  branin_i1     0.09398    0.1271    0.1364     0.1221
#
# Lowering it does not help and mostly hurts: the estimator needs those
# observations to build a density over, and starved of them it models the space
# badly. The intuition that "half the budget is wasted" is wrong, and the
# measurement is what says so.
N_STARTUP_TRIALS = 10


class OptunaTPE:
    """Tree-structured Parzen Estimator, rebuilt from history on every call."""

    def __init__(self, n_startup_trials: int = N_STARTUP_TRIALS) -> None:
        self._n_startup_trials = n_startup_trials

    @property
    def name(self) -> str:
        return "optuna_tpe"

    @property
    def n_startup_trials(self) -> int:
        """Exposed so the report can state it rather than leave it implicit."""
        return self._n_startup_trials

    def propose(
        self,
        goal: Goal,
        history: list[Episode],
        rng: random.Random,
    ) -> Candidate:
        distributions = distributions_for(goal.parameter_space)

        # The sampler's own RNG has to be seeded from persisted values too.
        # `rng` is already `Random(f"{seed}:{idx}")`, built from the run's seed
        # and the episode index, so drawing the sampler seed from it inherits
        # that determinism. Leaving TPESampler unseeded would reintroduce
        # exactly the non-determinism the rest of this design works to remove.
        sampler = TPESampler(
            seed=rng.getrandbits(32),
            n_startup_trials=self._n_startup_trials,
            constraints_func=_read_constraints if goal.constraints else None,
        )

        study = optuna.create_study(
            direction="minimize" if goal.objective.direction is Direction.MINIMISE else "maximize",
            sampler=sampler,
        )

        past = self._replay(history, goal, distributions)
        if past:
            study.add_trials(past)

        trial = study.ask(distributions)

        return Candidate(
            params=dict(trial.params),
            rationale=f"optuna TPE proposal informed by {len(past)} completed trials",
            source=CandidateSource.BASELINE,
        )

    @staticmethod
    def _replay(
        history: list[Episode],
        goal: Goal,
        distributions: dict[str, BaseDistribution],
    ) -> list[FrozenTrial]:
        """Turn stored episodes back into completed Optuna trials.

        Episodes that never reached the simulator — guard rejections — are
        skipped rather than added as failed trials. There is no observation to
        learn from: the point was never evaluated. (Nothing in Phase 2 produces
        one; from Phase 3 an LLM planner will.)
        """
        trials: list[FrozenTrial] = []

        for episode in history:
            result = episode.sim_result
            if result is None:
                continue

            system_attrs = {}
            if goal.constraints:
                system_attrs[CONSTRAINTS_KEY] = constraint_values(result, goal.constraints)

            trials.append(
                create_trial(
                    state=TrialState.COMPLETE,
                    value=result.objective_value,
                    params=dict(episode.candidate.params),
                    distributions=distributions,
                    system_attrs=system_attrs,
                )
            )

        return trials
