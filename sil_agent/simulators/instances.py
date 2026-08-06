"""Shifted and rotated benchmark instances, BBOB/COCO style.

Phase 3 discovered that the model has memorised the textbook benchmarks:
``SingleShotLLM`` proposed all three of Branin's global minima to four decimal
places *before seeing a single result*. Any comparison on such a function
measures recall, not search.

An instance fixes that by moving the problem somewhere the literature has never
written down, while keeping the one property that makes regret reportable — the
optimal *value* is unchanged, so "how far from the best possible" still means
something.

The construction
----------------

Work in normalised search coordinates ``z`` in the unit cube, and let
``f̂(w) = f(l + w * (u - l))`` be the original benchmark on normalised
coordinates. The instance is

    g(z) = f̂( R (z - s) )

with ``R`` a uniformly random rotation and ``s`` a shift, both from a seeded
RNG. That is the standard BBOB form, under which the optimum sits at
``s + R⁻¹w*``.

**This module inverts that relationship**, and the reason matters. Drawing ``s``
first puts the optimum wherever it happens to land — possibly outside the search
box, which would make the run unwinnable and regret meaningless. So the optimum
location is chosen first and the shift is solved for:

    z_opt = seeded draw from the interior of the cube
    s     = z_opt - Rᵀ w*                    (Rᵀ = R⁻¹ for a rotation)

Then ``g(z_opt) = f̂(w*) = f(x*) = known_optimum`` exactly, and the optimum is
inside the box by construction rather than by luck.

What this does not guarantee for free
-------------------------------------

The transform evaluates ``f`` *outside* its published domain, because a rotated
cube does not sit inside the original one. So ``known_optimum`` is only still
the optimum if ``f`` never dips lower out there. That is an assumption every
regret number would silently rest on, and it is checked by test rather than
argued — see ``tests/test_instances.py``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from sil_agent.agent.state import ParameterSpace, ParamKind, ParamSpec
from sil_agent.simulators.toy import ANONYMOUS_METRIC, Benchmark

# How far from the wall the optimum is placed, in normalised coordinates. Not
# cosmetic: an optimum at 0.02 is found by any strategy that samples a corner,
# and one exactly at the centre is found by any strategy that starts there.
# Keeping it in the interior measures search rather than either accident.
OPTIMUM_MARGIN = (0.15, 0.85)

# The optimum must not land on a round number, and this is not fussiness — it
# was caught destroying a measurement.
#
# The first `branin_i1` put its optimum at (0.6001, 0.4017). The agent proposed
# (0.5, 0.5), then (0.7, 0.3), then (0.6, 0.4), and scored regret 3e-4 in three
# evaluations on a function it had supposedly never seen. It had not recalled
# anything: it walked a coarse grid of round numbers and one of them *was* the
# answer.
#
# Language models propose round numbers overwhelmingly. Random search and TPE do
# not. So an optimum sitting on a grid point is not merely a lucky instance —
# it is differentially easy for exactly the strategy under test, which is a
# systematic bias in favour of the thing being measured.
#
# Every coordinate is therefore kept clear of the grid a model actually proposes
# on. Rejection sampling from the same seeded stream, so instances stay
# deterministic; only the draw count changes.
GRID_STEP = 0.05
GRID_CLEARANCE = 0.012
MAX_OPTIMUM_DRAWS = 1000

# Every instance is posed on the unit cube with neutral parameter names. This is
# half of the anonymisation, and the half that could not be achieved by renaming
# alone: a 2-D problem over exactly [-5, 10] x [0, 15] *is* Branin to anything
# that has read the literature, whatever the parameters are called.
INSTANCE_GOAL_TEXT = (
    "Minimise `objective` over the parameters below. The function is unknown: "
    "no analytic form, no published optimum, and no assumption of smoothness, "
    "separability or unimodality is available to you."
)


def _rotation(dim: int, rng: random.Random) -> list[list[float]]:
    """A uniformly random (Haar) rotation matrix, as nested lists.

    QR of a Gaussian matrix gives an orthogonal ``Q``, but not a *uniformly*
    distributed one — the signs of ``R``'s diagonal bias it. Multiplying each
    column of ``Q`` by the sign of the corresponding diagonal entry removes the
    bias. Without that correction some rotations are systematically more likely
    than others, which would make "randomly rotated" quietly untrue.

    ``PCG64`` is named explicitly rather than using ``default_rng``'s choice.
    The default is documented as changeable, and an instance whose geometry
    depends on the installed NumPy version is not reproducible in the sense this
    project claims.
    """
    generator = np.random.Generator(np.random.PCG64(rng.getrandbits(63)))
    matrix = generator.standard_normal((dim, dim))
    q, r = np.linalg.qr(matrix)
    corrected = q * np.sign(np.diag(r))
    return [[float(value) for value in row] for row in corrected]


def _distance_to_grid(value: float) -> float:
    """How far ``value`` sits from the nearest multiple of ``GRID_STEP``."""
    nearest = round(value / GRID_STEP) * GRID_STEP
    return abs(value - nearest)


def _draw_optimum_location(dim: int, rng: random.Random) -> tuple[float, ...]:
    """An interior point that is not a round number in any coordinate.

    Rejection sampling rather than nudging an offending coordinate: nudging
    would pile probability up just outside the exclusion zone, so optima would
    cluster at a fixed distance from the grid — a different artefact in place of
    the one being removed.
    """
    for _ in range(MAX_OPTIMUM_DRAWS):
        candidate = tuple(rng.uniform(*OPTIMUM_MARGIN) for _ in range(dim))
        if all(_distance_to_grid(value) >= GRID_CLEARANCE for value in candidate):
            return candidate

    # Unreachable with the configured constants (acceptance is ~52% per
    # coordinate), but a silent fallback to a grid-aligned optimum is exactly
    # the bug this function exists to prevent.
    raise RuntimeError(
        f"could not place an optimum clear of the {GRID_STEP} grid in {dim} dimensions "
        f"after {MAX_OPTIMUM_DRAWS} draws; loosen GRID_CLEARANCE"
    )


def _bounds_of(space: ParameterSpace) -> tuple[list[float], list[float]]:
    low: list[float] = []
    high: list[float] = []
    for spec in space.params:
        if spec.bounds is None:
            raise ValueError(
                f"cannot build an instance of a space with a non-numeric parameter: {spec.name}"
            )
        low.append(float(spec.bounds[0]))
        high.append(float(spec.bounds[1]))
    return low, high


def _unit_space(dim: int) -> ParameterSpace:
    return ParameterSpace(
        params=[
            ParamSpec(name=f"p{i}", kind=ParamKind.FLOAT, bounds=(0.0, 1.0))
            for i in range(1, dim + 1)
        ]
    )


def make_instance(base: Benchmark, instance_seed: int) -> Benchmark:
    """Build a shifted, rotated, anonymous instance of ``base``.

    The returned Benchmark is an ordinary one — same dataclass, same protocol —
    so every strategy, the guard, the harness and the report treat it exactly
    like any other. Nothing downstream knows it was transformed.
    """
    if base.constraints:
        # A constraint is stated on raw parameters, and rotating the space
        # rotates what the constraint means. Worse, `branin_constrained`'s
        # "the constrained optimum is still 0.397887" property depends on
        # exactly which of the three minima survive the half-plane, which does
        # not survive the transform. Doing this properly means transforming the
        # constraint and re-deriving the feasible optimum; refusing is honest.
        raise ValueError(
            f"{base.name} has constraints; shifted constrained instances are not supported "
            "(the constraint would have to be transformed and its optimum re-derived)"
        )

    dim = len(base.space.params)
    low, high = _bounds_of(base.space)

    # Seeded from the benchmark name as well as the number, so `branin_i1` and
    # `hartmann6_i1` are unrelated rather than sharing a rotation.
    rng = random.Random(f"{base.name}:{instance_seed}")

    rotation = _rotation(dim, rng)
    optimum_location = _draw_optimum_location(dim, rng)

    # A published optimiser, in normalised coordinates.
    optimiser = base.known_optimisers[0]
    w_star = [(optimiser[i] - low[i]) / (high[i] - low[i]) for i in range(dim)]

    # s = z_opt - Rᵀ w*, where (Rᵀ w)_i = sum_k R[k][i] w[k].
    shift = [
        optimum_location[i] - sum(rotation[k][i] * w_star[k] for k in range(dim))
        for i in range(dim)
    ]

    return replace(
        base,
        name=f"{base.name}_i{instance_seed}",
        space=_unit_space(dim),
        objective_metric=ANONYMOUS_METRIC,
        objective=_transformed_objective(base.objective, rotation, shift, low, high),
        known_optimisers=(optimum_location,),
        goal_text=INSTANCE_GOAL_TEXT,
        instance_seed=instance_seed,
        base_name=base.name,
        # known_optimum is deliberately inherited unchanged. That is the whole
        # point of the construction, and the test suite verifies it holds.
    )


def _transformed_objective(
    objective: Callable[[list[float]], float],
    rotation: Sequence[Sequence[float]],
    shift: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
) -> Callable[[list[float]], float]:
    """``z -> f(denormalise(R(z - s)))``, in plain Python.

    Closing over lists rather than NumPy arrays: this runs once per simulator
    call and the arrays would be built and torn down every time for a matrix
    multiply of at most six dimensions.
    """
    dim = len(shift)

    def transformed(z: list[float]) -> float:
        centred = [z[i] - shift[i] for i in range(dim)]
        rotated = [sum(rotation[i][j] * centred[j] for j in range(dim)) for i in range(dim)]
        original = [low[i] + rotated[i] * (high[i] - low[i]) for i in range(dim)]
        return objective(original)

    return transformed
