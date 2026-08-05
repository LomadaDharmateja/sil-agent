"""The simulator interface.

``Simulator`` is a ``typing.Protocol`` — structural typing. Unlike inheriting
from a base class, a Protocol requires no import and no registration: any object
that happens to have ``describe``, ``run`` and ``cost_per_eval`` with matching
signatures *is* a Simulator as far as mypy is concerned. That is what lets
``ToySimulator`` today and ``FMUSimulator`` in Phase 9 be swapped without either
one knowing the other exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from sil_agent.agent.state import Frozen, ParameterSpace, SimResult


class EvalCost(Frozen):
    """What one evaluation costs.

    Phase 2's ablation gives every strategy the same number of simulator calls;
    this is how a strategy can also be scored on wall-clock or money when the
    simulator stops being instant.
    """

    seconds: float
    euros: float = 0.0


class Simulator(Protocol):
    """The oracle. Produces every number the system treats as truth."""

    @property
    def name(self) -> str:
        """Stable identifier, used in run records and reports."""
        ...

    def describe(self) -> ParameterSpace:
        """The parameters this simulator accepts.

        The single source of truth for what exists. ``validate()`` checks
        proposals against this, which is why a model cannot invent a parameter.
        """
        ...

    def run(self, params: Mapping[str, float | int | str]) -> SimResult:
        """Evaluate one candidate. Assumes ``params`` has already passed the guard.

        ``Mapping`` rather than ``dict`` because ``dict`` is *invariant* in its
        value type: a ``dict[str, float]`` is not acceptable where a
        ``dict[str, float | int | str]`` is expected, even though every value in
        it would be. ``Mapping`` is covariant in its values, so a caller holding
        a narrower dictionary can pass it directly. It also documents that the
        simulator does not modify what it is given.
        """
        ...

    @property
    def cost_per_eval(self) -> EvalCost: ...
