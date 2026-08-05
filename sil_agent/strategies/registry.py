"""Building a strategy from its stored name.

This exists because of resume. A run records `strategy: "grid_search"` and
nothing else, so continuing it means reconstructing the strategy object from
that string plus whatever else is in the database — never from what the user
happened to type on the command line.

Grid search is why the context carries a budget. It has to know how many
evaluations it is allowed in order to size its grid, and if that number came
from an argument the user re-supplied at resume time, a run resumed with a
different `--episodes` would silently switch to a different grid and produce a
sequence that does not continue the original. `max_evaluations` lives in
`BudgetState`, which is persisted, so passing it here keeps the loop a pure
function of stored state.

**The router is built lazily.** Only the LLM strategies ask for one, so running
a baseline on a machine with no API key configured works exactly as it did in
Phase 2. Constructing the router eagerly would have made every baseline run
depend on a key it never uses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sil_agent.services.router import ModelRouter
from sil_agent.strategies.base import Strategy
from sil_agent.strategies.grid import GridSearch
from sil_agent.strategies.llm_agent import AgentNoReflection, SingleShotLLM
from sil_agent.strategies.optuna_tpe import OptunaTPE
from sil_agent.strategies.random_search import RandomSearch


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy might need, all of it derived from persisted state.

    ``router_factory`` is a callable rather than a router so that nothing is
    constructed — and no API key is required — until a strategy actually asks.
    """

    max_evaluations: int
    run_id: UUID | None = None
    router_factory: Callable[[], ModelRouter] | None = None

    def router(self) -> ModelRouter:
        if self.router_factory is None:
            raise RuntimeError(
                "This strategy needs an LLM router, but none was configured. "
                "Set CEREBRAS_API_KEY or GEMINI_API_KEY in .env (see .env.example)."
            )
        return self.router_factory()


# name -> factory. Phase 4 adds "agent_full" here, and nothing else in the
# harness, the CLI or the report has to change — which is the point of routing
# every strategy through one protocol.
STRATEGY_FACTORIES: dict[str, Callable[[StrategyContext], Strategy]] = {
    "random_search": lambda _: RandomSearch(),
    "grid_search": lambda ctx: GridSearch(ctx.max_evaluations),
    "optuna_tpe": lambda _: OptunaTPE(),
    "single_shot_llm": lambda ctx: SingleShotLLM(
        ctx.router(), max_evaluations=ctx.max_evaluations
    ),
    "agent_no_reflection": lambda ctx: AgentNoReflection(
        ctx.router(), max_evaluations=ctx.max_evaluations
    ),
}

STRATEGY_NAMES: list[str] = sorted(STRATEGY_FACTORIES)

# The subset that needs a provider. Used by the harness and CLI to fail early
# with a clear message rather than partway through a matrix.
LLM_STRATEGIES: frozenset[str] = frozenset({"single_shot_llm", "agent_no_reflection"})


def build_strategy(
    name: str,
    *,
    max_evaluations: int,
    run_id: UUID | None = None,
    router_factory: Callable[[], ModelRouter] | None = None,
) -> Strategy:
    """Construct a strategy by name, or raise KeyError listing what exists."""
    factory = STRATEGY_FACTORIES.get(name)
    if factory is None:
        raise KeyError(f"unknown strategy {name!r}; available: {', '.join(STRATEGY_NAMES)}")

    return factory(
        StrategyContext(
            max_evaluations=max_evaluations,
            run_id=run_id,
            router_factory=router_factory,
        )
    )


def needs_llm(name: str) -> bool:
    return name in LLM_STRATEGIES
