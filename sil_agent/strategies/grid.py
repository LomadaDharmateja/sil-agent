"""Grid search — naive systematic coverage.

The algorithm is the least interesting thing in this file. Two other things are
worth reading it for.

**Sizing the grid to the budget.** With ``d`` parameters and a budget of ``N``
evaluations, each parameter gets ``k = floor(N ** (1/d))`` points, so the whole
grid is ``k ** d <= N`` and always completes. For Branin at N=200 that is 14x14
= 196 points. For Hartmann-6 at the same budget it is 2**6 = 64 — grid search
covers its entire grid using a third of its allowance and then stops, because
two points per axis is all a 200-evaluation budget buys in six dimensions.

That is not a defect to engineer around. It is the curse of dimensionality
turning up on the comparison table, and it is one of the more useful things the
table has to say. What matters is that the report distinguishes "stopped with
budget left because it finished" from "used everything it had" — hence
:class:`StrategyExhausted` and ``TerminationReason.EXHAUSTED``.

The rejected alternative is worth recording because it looks more thorough and
is actually much worse: build a grid *larger* than the budget and let the run
truncate it. Enumeration varies the last parameter fastest, so a grid truncated
at 200 of 4096 points leaves the first parameter pinned at its lower bound for
the entire run. The result is labelled "grid search" but is really a line search
along one axis, and it would lose the comparison for a reason that has nothing
to do with grid search.

**No counter on the object.** The position in the enumeration is derived from
``history`` on every call. A ``self._next_index`` would work perfectly until a
run was resumed in a fresh process, at which point the grid would restart from
its first point and re-evaluate everything already done — the same class of bug
as a strategy holding its own RNG, and equally invisible without a resume test.
"""

from __future__ import annotations

import math
import random

from sil_agent.agent.state import (
    Candidate,
    CandidateSource,
    Episode,
    Goal,
    ParameterSpace,
    ParamKind,
    ParamSpec,
)
from sil_agent.strategies.base import StrategyExhausted


def axis_values(spec: ParamSpec, points: int) -> list[float | int | str]:
    """The values one parameter takes across the grid.

    Float axes are ``points`` evenly spaced values **including both bounds**.
    Endpoints rather than cell centres is a real choice with visible
    consequences: Rosenbrock's optimum sits at x=1, and a 3-point grid over
    [-5, 10] visits -5, 2.5 and 10, so it never gets close. Cell centres would
    miss it differently rather than less. Endpoints are what a person drawing a
    grid by hand does, and being predictable matters more here than being
    lucky.

    Integer axes are the same values rounded, then de-duplicated — asking for 10
    points on an axis spanning [0, 3] yields 4, not 10 with repeats.

    Categorical axes ignore ``points`` and take every choice. There is no
    meaningful way to sample two thirds of a set of names.
    """
    if spec.kind is ParamKind.CATEGORICAL:
        assert spec.choices is not None  # guaranteed by ParamSpec validation
        return list(spec.choices)

    assert spec.bounds is not None  # guaranteed by ParamSpec validation
    low, high = spec.bounds

    if points < 2:
        # A single point goes in the middle. An axis pinned to its lower bound
        # would be a strictly worse use of the one sample available.
        centre = (low + high) / 2.0
        if spec.kind is ParamKind.INT:
            return [round(centre)]
        return [centre]

    step = (high - low) / (points - 1)
    raw = [low + step * i for i in range(points)]

    if spec.kind is ParamKind.INT:
        # Round inward first so a rounded endpoint cannot leave the declared
        # range, then de-duplicate while preserving order.
        integers = [min(max(round(value), math.ceil(low)), math.floor(high)) for value in raw]
        unique: list[float | int | str] = []
        for value in integers:
            if value not in unique:
                unique.append(value)
        return unique

    return list(raw)


def points_per_axis(space: ParameterSpace, max_evaluations: int) -> int:
    """How many points each numeric axis gets, so the whole grid fits the budget.

    Categorical axes are not negotiable — they contribute ``len(choices)``
    whatever happens — so their product is divided out of the budget first and
    the numeric axes share what is left.
    """
    numeric = [spec for spec in space.params if spec.kind is not ParamKind.CATEGORICAL]
    categorical_size = 1
    for spec in space.params:
        if spec.kind is ParamKind.CATEGORICAL:
            assert spec.choices is not None
            categorical_size *= len(spec.choices)

    if not numeric:
        return 1

    available = max_evaluations / categorical_size
    if available < 1:
        return 1

    points = int(available ** (1.0 / len(numeric)))

    # Floating-point exponentiation lands just below an exact answer often
    # enough to matter: 196 ** 0.5 can come out as 13.999999999999998, which
    # truncates to 13 and quietly wastes 27 evaluations. Step up while the
    # larger grid still fits.
    while (points + 1) ** len(numeric) * categorical_size <= max_evaluations:
        points += 1

    return max(1, points)


class GridSearch:
    """Enumerates a grid sized to fit the evaluation budget.

    The budget is a constructor argument because ``Strategy.propose`` is not
    given one — it receives the goal, the history and an RNG. It is passed in
    from ``BudgetState.max_evaluations``, which is persisted, so a resumed run
    reconstructs an identical grid. See ``strategies/registry.py``.
    """

    def __init__(self, max_evaluations: int) -> None:
        if max_evaluations < 1:
            raise ValueError(f"max_evaluations must be at least 1, got {max_evaluations}")
        self._max_evaluations = max_evaluations

    @property
    def name(self) -> str:
        return "grid_search"

    def axes(self, space: ParameterSpace) -> list[list[float | int | str]]:
        """The value list for each parameter, in the space's declared order."""
        points = points_per_axis(space, self._max_evaluations)
        return [axis_values(spec, points) for spec in space.params]

    def total_points(self, space: ParameterSpace) -> int:
        total = 1
        for values in self.axes(space):
            total *= len(values)
        return total

    def point_at(self, space: ParameterSpace, index: int) -> dict[str, float | int | str]:
        """The ``index``-th grid point, counting from 0.

        Plain odometer arithmetic: the last axis advances every step, the one
        before it every ``len(last)`` steps, and so on. Computing the point from
        its index rather than iterating to it is what lets the position be
        derived from history instead of remembered.
        """
        axes = self.axes(space)
        remaining = index
        params: dict[str, float | int | str] = {}

        for spec, values in zip(reversed(space.params), reversed(axes), strict=True):
            params[spec.name] = values[remaining % len(values)]
            remaining //= len(values)

        # Rebuilt in declared order; the loop above walked the axes backwards.
        return {spec.name: params[spec.name] for spec in space.params}

    def propose(
        self,
        goal: Goal,
        history: list[Episode],
        rng: random.Random,
    ) -> Candidate:
        """The next unvisited grid point. ``rng`` is unused — this is deterministic."""
        space = goal.parameter_space

        # Derived, never remembered. `max(idx) + 1` rather than `len(history)`
        # for the same reason the loop uses it: if a gap ever appeared, this
        # keeps moving forward instead of re-proposing a point already tried.
        index = max((episode.idx for episode in history), default=-1) + 1

        total = self.total_points(space)
        if index >= total:
            raise StrategyExhausted(
                f"grid of {total} points is fully covered after {index} episodes"
            )

        return Candidate(
            params=self.point_at(space, index),
            rationale=f"grid point {index + 1} of {total}",
            source=CandidateSource.BASELINE,
        )
