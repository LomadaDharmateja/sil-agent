"""The guard — deterministic code that stands between a proposal and the oracle.

In Phase 1 the only thing proposing candidates is a uniform random sampler,
which by construction cannot produce a bad one. The guard is built properly now
anyway, because from Phase 3 the proposer is an LLM and this is the component
that makes Rule 2 real: *the LLM proposes, deterministic code disposes.*

The policy, and why each choice is what it is:

======================  ==========  ==================================================
Situation               Response    Reasoning
======================  ==========  ==================================================
Unknown parameter name  reject      A hallucinated parameter is a reasoning failure.
                                    Silently dropping it hides the failure.
Missing parameter       reject      Filling in a default would invent a design
                                    decision and attribute it to the model.
Value out of bounds     clamp       Being 5% outside a range is a near-miss worth
                                    evaluating. Clamping is recorded, not silent.
float 3.0 for an INT    coerce      JSON has one number type; this is a
                                    serialisation artefact, not an error.
float 3.7 for an INT    reject      Rounding would hide a genuine misunderstanding
                                    of the parameter.
Numeric string "3.5"    coerce      Same serialisation artefact, recorded.
Bad categorical value   reject      There is no sensible nearest choice.
NaN or infinity         reject      Would silently poison every later comparison.
======================  ==========  ==================================================

One case is handled one layer earlier, in ``Candidate`` itself: a boolean value.
``bool`` subclasses ``int``, so ``True`` is absorbed by the ``float | int | str``
union and stored as ``1.0`` before the guard is ever called. It is rejected
during model validation instead, where the evidence still exists.
"""

from __future__ import annotations

import math

from sil_agent.agent.state import Candidate, Frozen, ParameterSpace, ParamKind, ParamSpec


class GuardRejection(Exception):
    """A candidate could not be repaired into something the simulator can run.

    Raised rather than returned: a rejected candidate has no valid form, so
    there is nothing sensible for the caller to do with a return value. The loop
    catches this and records the episode as a ToolError, which keeps the failure
    in the permanent history instead of losing it.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class GuardResult(Frozen):
    """The cleaned candidate plus a record of what had to be changed.

    ``clamped`` and ``coerced`` are deliberately surfaced rather than hidden:
    from Phase 3 they are evidence about the proposer's quality. A planner whose
    values are constantly clamped is a planner that has not understood the
    parameter space, and that is worth being able to see.
    """

    candidate: Candidate
    clamped: list[str]
    coerced: list[str]

    @property
    def was_modified(self) -> bool:
        return bool(self.clamped or self.coerced)


def validate(candidate: Candidate, space: ParameterSpace) -> GuardResult:
    """Check a candidate against the space the simulator declares.

    Returns a new Candidate with cleaned values. Raises GuardRejection if the
    candidate cannot be repaired.
    """
    proposed = set(candidate.params)
    declared = set(space.names)

    unknown = sorted(proposed - declared)
    if unknown:
        raise GuardRejection(
            f"unknown parameter(s): {', '.join(unknown)}. "
            f"Declared parameters are: {', '.join(space.names)}"
        )

    missing = sorted(declared - proposed)
    if missing:
        raise GuardRejection(f"missing required parameter(s): {', '.join(missing)}")

    cleaned: dict[str, float | int | str] = {}
    clamped: list[str] = []
    coerced: list[str] = []

    for spec in space.params:
        raw = candidate.params[spec.name]
        value, was_coerced, was_clamped = _clean_one(spec, raw)
        cleaned[spec.name] = value
        if was_coerced:
            coerced.append(spec.name)
        if was_clamped:
            clamped.append(spec.name)

    return GuardResult(
        candidate=Candidate(
            params=cleaned,
            rationale=candidate.rationale,
            source=candidate.source,
        ),
        clamped=clamped,
        coerced=coerced,
    )


def _clean_one(spec: ParamSpec, raw: float | int | str) -> tuple[float | int | str, bool, bool]:
    """Clean a single value. Returns (value, was_coerced, was_clamped)."""
    if spec.kind is ParamKind.CATEGORICAL:
        return _clean_categorical(spec, raw), False, False
    if spec.kind is ParamKind.INT:
        return _clean_int(spec, raw)
    return _clean_float(spec, raw)


def _clean_categorical(spec: ParamSpec, raw: float | int | str) -> str:
    choices = spec.choices or []
    if not isinstance(raw, str):
        raise GuardRejection(
            f"{spec.name}: expected one of {choices}, got {type(raw).__name__} {raw!r}"
        )
    if raw not in choices:
        raise GuardRejection(f"{spec.name}: {raw!r} is not one of {choices}")
    return raw


def _clean_float(spec: ParamSpec, raw: float | int | str) -> tuple[float, bool, bool]:
    value, was_coerced = _as_number(spec, raw)
    number = float(value)

    assert spec.bounds is not None  # guaranteed by ParamSpec validation
    low, high = spec.bounds
    clamped_value = min(max(number, low), high)
    return clamped_value, was_coerced, clamped_value != number


def _clean_int(spec: ParamSpec, raw: float | int | str) -> tuple[int, bool, bool]:
    value, was_coerced = _as_number(spec, raw)

    if isinstance(value, float):
        if not value.is_integer():
            raise GuardRejection(
                f"{spec.name}: expected an integer, got {value!r}. "
                "Rounding would hide a misunderstanding of this parameter."
            )
        was_coerced = True
    number = int(value)

    assert spec.bounds is not None  # guaranteed by ParamSpec validation
    low, high = spec.bounds
    # Round the bounds inward so clamping can never land outside the range.
    clamped_value = min(max(number, math.ceil(low)), math.floor(high))
    return clamped_value, was_coerced, clamped_value != number


def _as_number(spec: ParamSpec, raw: float | int | str) -> tuple[float | int, bool]:
    """Coerce to a number, or reject. Returns (value, was_coerced)."""
    was_coerced = False

    if isinstance(raw, str):
        try:
            parsed: float | int = float(raw)
        except ValueError:
            raise GuardRejection(f"{spec.name}: {raw!r} is not a number") from None
        raw = parsed
        was_coerced = True

    if isinstance(raw, float) and not math.isfinite(raw):
        raise GuardRejection(f"{spec.name}: {raw!r} is not a finite number")

    return raw, was_coerced
