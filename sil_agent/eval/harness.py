"""Executing an ablation: every strategy, on every benchmark, at every seed.

The harness is deliberately thin. It works out which runs an experiment
consists of, makes sure each one exists, and hands each to ``run_loop``. It
contains no scoring, no plotting and no knowledge of what any strategy does.

**The one idea in this file is deterministic run identity.**

Each cell of the matrix gets a ``run_id`` derived with UUIDv5 from
``(experiment, strategy, simulator, seed, max_evaluations)``. UUIDv5 hashes a
name into a UUID, so the same tuple always yields the same identifier — the id
is a *function of the configuration* rather than something freshly invented.

Everything else follows from that:

* **Restart by re-running the same command.** Each cell is either absent
  (create it), present but unfinished (load, rehydrate, continue), or finished
  (skip). Phase 1's idempotent ``append_episode`` and recompute-on-load supply
  the rest. There is no job table, no progress file and no ``--resume`` flag,
  because there is nothing to keep track of.

* **Accidental duplication is impossible.** Running the matrix twice cannot
  produce two sets of runs for the same configuration and quietly average four
  seeds' data into a five-seed cell — the second attempt lands on ids that
  already exist.

* **The identity is reproducible on another machine**, so an experiment can be
  described by its spec alone.

The cost is that changing the budget changes every id, so a 200-evaluation
experiment and a 400-evaluation one never collide. That is the intended
behaviour: they are different experiments and their runs must not be pooled.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid5

from sil_agent.agent.loop import LoopResult, rehydrate, run_loop
from sil_agent.agent.state import BudgetState, RunState, RunStatus, utcnow
from sil_agent.eval.metrics import derive_termination
from sil_agent.persistence.repo import RunRepository
from sil_agent.services.replay import CachingRouter
from sil_agent.services.router import ModelRouter, build_default_router
from sil_agent.simulators.toy import ToySimulator
from sil_agent.strategies.registry import build_strategy

# A fixed, arbitrary namespace UUID. Its only job is to keep these identifiers
# from colliding with UUIDv5s generated elsewhere for other purposes. It must
# never change: changing it renames every run in every experiment ever executed,
# and old results become unreachable.
EXPERIMENT_NAMESPACE = UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


@dataclass(frozen=True)
class Cell:
    """One configuration in the matrix — a single run."""

    experiment: str
    strategy: str
    simulator: str
    seed: int
    max_evaluations: int

    @property
    def run_id(self) -> UUID:
        """The deterministic identity of this run.

        Field order and separator are part of the contract: any change to how
        this string is built silently orphans every run recorded before it.
        """
        name = "|".join(
            [
                self.experiment,
                self.strategy,
                self.simulator,
                str(self.seed),
                str(self.max_evaluations),
            ]
        )
        return uuid5(EXPERIMENT_NAMESPACE, name)

    @property
    def label(self) -> str:
        return f"{self.strategy}/{self.simulator}/seed={self.seed}"


@dataclass(frozen=True)
class ExperimentSpec:
    """The definition of an ablation.

    Every strategy gets the same simulators, the same seeds and the same
    evaluation budget. That is the fairness rule the comparison rests on, and
    expressing the matrix as a product rather than a list of runs is what makes
    it impossible to violate by accident.
    """

    name: str
    strategies: Sequence[str]
    simulators: Sequence[str]
    seeds: Sequence[int]
    max_evaluations: int
    max_rejections: int = field(default=50)

    def cells(self) -> Iterator[Cell]:
        """Every run in the experiment. Strategy-major, so partial results are comparable."""
        for strategy in self.strategies:
            for simulator in self.simulators:
                for seed in self.seeds:
                    yield Cell(
                        experiment=self.name,
                        strategy=strategy,
                        simulator=simulator,
                        seed=seed,
                        max_evaluations=self.max_evaluations,
                    )

    @property
    def total_runs(self) -> int:
        return len(self.strategies) * len(self.simulators) * len(self.seeds)


def _router_factory(run_id: UUID, repo: RunRepository) -> Callable[[], ModelRouter]:
    """A router that records every call against this run and replays repeats.

    Returned as a factory rather than a router so that nothing is constructed —
    and no API key is needed — unless a strategy actually asks for one. A matrix
    of baselines runs on a machine with no key configured.
    """

    def build() -> ModelRouter:
        return CachingRouter(build_default_router(), repo, run_id)

    return build


@dataclass(frozen=True)
class CellOutcome:
    """What happened to one cell during this invocation."""

    cell: Cell
    status: str  # "skipped" | "completed"
    episodes_run: int
    result: LoopResult | None


def ensure_run(cell: Cell, spec: ExperimentSpec, repo: RunRepository) -> RunState:
    """Load the run for this cell, creating it if it does not exist yet.

    The run row must exist before any episode, because episodes reference it by
    foreign key — and creating it up front is also what makes a cell resumable
    from the very first instant.
    """
    stored = repo.load_run(cell.run_id)
    if stored is not None:
        # rehydrate() recomputes position, best and budget from the episodes
        # table. Whatever the snapshot columns say, the episodes are the record.
        return rehydrate(stored.state)

    simulator = ToySimulator.from_name(cell.simulator)
    now = utcnow()
    state = RunState(
        run_id=cell.run_id,
        goal=simulator.default_goal(),
        status=RunStatus.PENDING,
        history=[],
        best=None,
        budget=BudgetState(
            max_evaluations=cell.max_evaluations,
            max_rejections=spec.max_rejections,
        ),
        step_idx=0,
        seed=cell.seed,
        created_at=now,
        updated_at=now,
    )
    repo.save_run(
        state,
        simulator=cell.simulator,
        strategy=cell.strategy,
        experiment=cell.experiment,
    )
    return state


def execute_cell(cell: Cell, spec: ExperimentSpec, repo: RunRepository) -> CellOutcome:
    """Run one cell to completion, or skip it if it already finished."""
    stored = repo.load_run(cell.run_id)
    if stored is not None and derive_termination(stored) is not None:
        return CellOutcome(cell=cell, status="skipped", episodes_run=0, result=None)

    state = ensure_run(cell, spec, repo)
    simulator = ToySimulator.from_name(cell.simulator)

    # The budget comes from the loaded state rather than from the spec: on
    # resume, the persisted value is the authority. They agree unless the spec
    # was edited between invocations, in which case the run's own budget — the
    # one its grid and its history were built against — is the correct one.
    strategy = build_strategy(
        cell.strategy,
        max_evaluations=state.budget.max_evaluations,
        run_id=cell.run_id,
        # Every model call is recorded against this run and replayed on a second
        # invocation, so re-running a finished matrix costs nothing and a
        # resumed run reuses the answers it already paid for.
        router_factory=_router_factory(cell.run_id, repo),
    )

    result = run_loop(state=state, simulator=simulator, strategy=strategy, repo=repo)

    return CellOutcome(
        cell=cell,
        status="completed",
        episodes_run=result.episodes_run,
        result=result,
    )


def execute(
    spec: ExperimentSpec,
    repo: RunRepository,
    *,
    verbose: bool = True,
) -> list[CellOutcome]:
    """Execute every cell of the experiment, in order.

    Sequential on purpose. The simulators are microseconds and the bottleneck is
    two database round trips per episode, so parallelism would buy a few minutes
    at the cost of concurrent writers — which the loop does not yet guard
    against (Redis locking is Phase 3). An ablation that takes twenty minutes
    and is obviously correct beats one that takes five and needs an argument.
    """
    outcomes: list[CellOutcome] = []
    total = spec.total_runs

    for index, cell in enumerate(spec.cells(), start=1):
        outcome = execute_cell(cell, spec, repo)
        outcomes.append(outcome)

        if verbose:
            if outcome.status == "skipped":
                detail = "already complete"
            else:
                assert outcome.result is not None
                best = outcome.result.state.best
                score = f"{best.result.objective_value:.6f}" if best else "none"
                detail = (
                    f"{outcome.result.reason.value:<10} "
                    f"episodes={outcome.episodes_run:<4} best={score}"
                )
            # flush=True for the same reason as everywhere else in this project:
            # redirected stdout is buffered, and a killed process flushes
            # nothing. An interrupted matrix must leave a usable log.
            print(f"[{index:>3}/{total}] {cell.label:<45} {detail}", flush=True)

    return outcomes
