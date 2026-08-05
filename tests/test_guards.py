"""The guard.

In Phase 1 nothing exercises these paths in anger — the random sampler cannot
produce an out-of-bounds value or a hallucinated parameter name. These tests
exist because in Phase 3 an LLM starts proposing candidates and this is the
component standing between it and the oracle. The behaviour is pinned down now,
while it is easy to reason about.
"""

from __future__ import annotations

import math

import pytest

from sil_agent.agent.guards import GuardRejection, validate
from sil_agent.agent.state import (
    Candidate,
    CandidateSource,
    ParameterSpace,
    ParamKind,
    ParamSpec,
)

SPACE = ParameterSpace(
    params=[
        ParamSpec(name="ratio", kind=ParamKind.FLOAT, bounds=(0.0, 1.0)),
        ParamSpec(name="cells", kind=ParamKind.INT, bounds=(1.0, 10.0)),
        ParamSpec(name="material", kind=ParamKind.CATEGORICAL, choices=["steel", "alu"]),
    ]
)


def candidate(**params: float | int | str) -> Candidate:
    return Candidate(params=params, rationale="test", source=CandidateSource.PLANNER)


def valid(**overrides: float | int | str) -> Candidate:
    params: dict[str, float | int | str] = {
        "ratio": 0.5,
        "cells": 5,
        "material": "steel",
    }
    params.update(overrides)
    return candidate(**params)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_valid_candidate_passes_through_unchanged() -> None:
    result = validate(valid(), SPACE)
    assert result.candidate.params == {"ratio": 0.5, "cells": 5, "material": "steel"}
    assert not result.was_modified


def test_metadata_is_preserved() -> None:
    """The guard cleans values; it must not rewrite the proposer's reasoning."""
    result = validate(valid(), SPACE)
    assert result.candidate.rationale == "test"
    assert result.candidate.source is CandidateSource.PLANNER


# ---------------------------------------------------------------------------
# Rejection: unknown and missing names
# ---------------------------------------------------------------------------


def test_unknown_parameter_is_rejected() -> None:
    """The anti-hallucination guard. Dropping the key silently would hide it."""
    with pytest.raises(GuardRejection, match="unknown parameter"):
        validate(valid(thickness=3.0), SPACE)


def test_rejection_message_lists_the_declared_parameters() -> None:
    """The message becomes LLM feedback in Phase 3, so it has to be actionable."""
    with pytest.raises(GuardRejection) as exc_info:
        validate(valid(thickness=3.0), SPACE)
    message = str(exc_info.value)
    assert "thickness" in message
    assert "ratio" in message and "cells" in message and "material" in message


def test_missing_parameter_is_rejected() -> None:
    with pytest.raises(GuardRejection, match="missing required parameter"):
        validate(candidate(ratio=0.5, cells=5), SPACE)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_float_above_the_upper_bound_is_clamped() -> None:
    result = validate(valid(ratio=1.7), SPACE)
    assert result.candidate.params["ratio"] == 1.0
    assert result.clamped == ["ratio"]


def test_float_below_the_lower_bound_is_clamped() -> None:
    result = validate(valid(ratio=-0.4), SPACE)
    assert result.candidate.params["ratio"] == 0.0
    assert result.clamped == ["ratio"]


def test_value_exactly_on_the_bound_is_not_clamped() -> None:
    result = validate(valid(ratio=1.0), SPACE)
    assert result.candidate.params["ratio"] == 1.0
    assert result.clamped == []


def test_int_is_clamped_within_its_bounds() -> None:
    result = validate(valid(cells=99), SPACE)
    assert result.candidate.params["cells"] == 10
    assert result.clamped == ["cells"]


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def test_integral_float_is_coerced_for_an_int_parameter() -> None:
    """JSON has one number type, so 3 often arrives as 3.0. Not an error."""
    result = validate(valid(cells=3.0), SPACE)
    assert result.candidate.params["cells"] == 3
    assert isinstance(result.candidate.params["cells"], int)
    assert result.coerced == ["cells"]


def test_non_integral_float_for_an_int_parameter_is_rejected() -> None:
    """Rounding 3.7 to 4 would hide a real misunderstanding of the parameter."""
    with pytest.raises(GuardRejection, match="expected an integer"):
        validate(valid(cells=3.7), SPACE)


def test_numeric_string_is_coerced() -> None:
    result = validate(valid(ratio="0.25"), SPACE)
    assert result.candidate.params["ratio"] == 0.25
    assert result.coerced == ["ratio"]


def test_non_numeric_string_is_rejected() -> None:
    with pytest.raises(GuardRejection, match="is not a number"):
        validate(valid(ratio="quite high"), SPACE)


def test_coerced_string_is_still_clamped() -> None:
    result = validate(valid(ratio="4.5"), SPACE)
    assert result.candidate.params["ratio"] == 1.0
    assert result.coerced == ["ratio"]
    assert result.clamped == ["ratio"]


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------


def test_unknown_choice_is_rejected() -> None:
    """There is no sensible nearest value for a category, so no repair is possible."""
    with pytest.raises(GuardRejection, match="is not one of"):
        validate(valid(material="titanium"), SPACE)


def test_number_for_a_categorical_is_rejected() -> None:
    with pytest.raises(GuardRejection, match="expected one of"):
        validate(valid(material=3), SPACE)


# ---------------------------------------------------------------------------
# Values that would poison later comparisons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(bad: float) -> None:
    """NaN compares False against everything, so it would silently break
    `better_than` and quietly freeze the incumbent forever."""
    with pytest.raises(GuardRejection, match="not a finite number"):
        validate(valid(ratio=bad), SPACE)


def test_boolean_is_rejected_before_the_guard_ever_sees_it() -> None:
    """bool subclasses int, so `float | int | str` absorbs True into 1.0.

    That happens during model validation, before the guard runs, so the
    rejection has to live there too — by the time `validate()` is called the
    boolean is already gone.
    """
    with pytest.raises(ValueError, match="boolean"):
        valid(cells=True)


# ---------------------------------------------------------------------------
# The parameter space itself
# ---------------------------------------------------------------------------


def test_categorical_without_choices_is_invalid() -> None:
    with pytest.raises(ValueError, match="needs choices"):
        ParamSpec(name="bad", kind=ParamKind.CATEGORICAL)


def test_float_without_bounds_is_invalid() -> None:
    with pytest.raises(ValueError, match="needs bounds"):
        ParamSpec(name="bad", kind=ParamKind.FLOAT)


def test_inverted_bounds_are_invalid() -> None:
    with pytest.raises(ValueError, match="low < high"):
        ParamSpec(name="bad", kind=ParamKind.FLOAT, bounds=(5.0, 1.0))


def test_duplicate_parameter_names_are_invalid() -> None:
    with pytest.raises(ValueError, match="unique"):
        ParameterSpace(
            params=[
                ParamSpec(name="x", kind=ParamKind.FLOAT, bounds=(0.0, 1.0)),
                ParamSpec(name="x", kind=ParamKind.FLOAT, bounds=(0.0, 1.0)),
            ]
        )
