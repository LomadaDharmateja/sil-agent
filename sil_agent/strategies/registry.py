"""Building a strategy from its stored name.

This exists because of resume. A run records `strategy: "grid_search"` and
nothing else, so continuing it means reconstructing the strategy object from
that string plus whatever else is in the database — never from what the user
happened to type on the command line.

Grid search is why the signature takes a budget. It has to know how many
evaluations it is allowed in order to size its grid, and if that number came
from an argument the user re-supplied at resume time, a run resumed with a
different `--episodes` would silently switch to a different grid and produce a
sequence that does not continue the original. `max_evaluations` lives in
`BudgetState`, which is persisted, so passing it here keeps the loop a pure
function of stored state.

Strategies that do not need it simply ignore it — the factory signature is
uniform so that callers never special-case.
"""

from __future__ import annotations

from collections.abc import Callable

from sil_agent.strategies.base import Strategy
from sil_agent.strategies.grid import GridSearch
from sil_agent.strategies.optuna_tpe import OptunaTPE
from sil_agent.strategies.random_search import RandomSearch

# name -> factory taking the evaluation budget.
#
# Phase 3 adds "llm_single" and Phase 4 "agent_no_reflection" / "agent_full"
# here. Nothing else in the harness, the CLI or the report has to change, which
# is the point of routing every strategy through one protocol.
STRATEGY_FACTORIES: dict[str, Callable[[int], Strategy]] = {
    "random_search": lambda _: RandomSearch(),
    "grid_search": lambda budget: GridSearch(budget),
    "optuna_tpe": lambda _: OptunaTPE(),
}

STRATEGY_NAMES: list[str] = sorted(STRATEGY_FACTORIES)


def build_strategy(name: str, *, max_evaluations: int) -> Strategy:
    """Construct a strategy by name, or raise KeyError listing what exists."""
    factory = STRATEGY_FACTORIES.get(name)
    if factory is None:
        raise KeyError(f"unknown strategy {name!r}; available: {', '.join(STRATEGY_NAMES)}")
    return factory(max_evaluations)
