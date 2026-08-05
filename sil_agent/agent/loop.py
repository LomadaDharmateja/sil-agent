"""The loop.

The structure is TECHNICAL_DESIGN §3 with the LLM steps stubbed: a strategy
proposes instead of a planner, and the critic and replanner are placeholders
carrying computed values and no prose.

Everything interesting in this file is about one property — **durable
execution**. Kill the process at any instant and a later ``resume`` continues
exactly where it left off, with no episode lost and none run twice.

Three things make that true:

1. **The episode insert is the commit point.** One statement decides whether an
   episode happened. Die before it and the episode is re-run; die after it and
   it is skipped. There is no in-between state to reconcile, because there is
   no moment at which an episode is half-recorded.

2. **Position is recomputed, never remembered.** ``step_idx``, ``best`` and the
   budget are all derived from the episodes table on load. The matching columns
   on ``runs`` exist for cheap inspection and are deliberately not trusted. If
   they ever disagree with the episodes, the episodes are right.

3. **Randomness is derived from persisted values.** Each episode's random
   number generator is seeded from ``(seed, idx)``, both of which are in the
   database. A generator that carried state across episodes would produce a
   different sequence after a restart, and "resume gives the identical run"
   would quietly become false.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sil_agent.agent.guards import GuardRejection, validate
from sil_agent.agent.state import (
    Best,
    BudgetState,
    CostRecord,
    Episode,
    Evaluation,
    Goal,
    Objective,
    ReplanDecision,
    RunState,
    RunStatus,
    SimResult,
    TerminationReason,
    ToolError,
)
from sil_agent.simulators.base import Simulator
from sil_agent.strategies.base import Strategy, StrategyExhausted


def episode_rng(seed: int, idx: int) -> random.Random:
    """The random number generator for one episode.

    Seeded from persisted values only, so it is identical however the run got
    here — first attempt, resumed after a crash, or replayed next week.

    Seeding with a string is deliberate: Python hashes a string seed with SHA-512
    internally, so the result is stable across processes and unaffected by
    ``PYTHONHASHSEED``. Seeding with ``seed + idx`` would work too but collides
    across runs (seed 1 episode 2 and seed 2 episode 1 would share a stream).
    """
    return random.Random(f"{seed}:{idx}")


def recompute_step_idx(history: list[Episode]) -> int:
    """The index of the next episode to run, derived from what is stored.

    Deliberately ``max(idx) + 1`` rather than ``len(history)``: they agree when
    the sequence is contiguous, but if a gap ever appeared, this keeps writing
    forward instead of overwriting an existing episode.
    """
    if not history:
        return 0
    return max(episode.idx for episode in history) + 1


def recompute_best(history: list[Episode], objective: Objective) -> Best | None:
    """Replay the history to find the best result. Never loaded from a cache.

    This is the same ``better_than`` comparison the live loop uses, applied in
    order, so a resumed run reaches exactly the incumbent the original had.
    """
    best: Best | None = None
    for episode in history:
        result = episode.sim_result
        if result is None:
            continue
        if result.better_than(best, objective):
            best = Best(episode_idx=episode.idx, candidate=episode.candidate, result=result)
    return best


def recompute_budget(budget: BudgetState, history: list[Episode]) -> BudgetState:
    """Rebuild budget usage from the episodes actually recorded.

    Only episodes that reached the simulator consume the evaluation budget; a
    candidate rejected by the guard never cost an evaluation, and is counted
    against the separate rejection allowance instead. Wall-clock and euro usage
    are summed from the same records.
    """
    evaluations = 0
    rejections = 0
    wall_clock = 0.0
    cost = 0.0
    for episode in history:
        result = episode.sim_result
        if result is not None:
            evaluations += 1
            wall_clock += result.wall_time_s
        else:
            rejections += 1
        cost += episode.cost.cost_eur

    return BudgetState(
        max_evaluations=budget.max_evaluations,
        evaluations_used=evaluations,
        max_cost_eur=budget.max_cost_eur,
        cost_eur_used=cost,
        max_wall_clock_s=budget.max_wall_clock_s,
        wall_clock_s_used=wall_clock,
        max_rejections=budget.max_rejections,
        rejections_used=rejections,
    )


def rehydrate(state: RunState) -> RunState:
    """Return the state with every derived field recomputed from history.

    Rule 1, made concrete. Call this after loading a run and before continuing
    it: whatever ``runs.step_idx`` and ``runs.best`` happen to say, this
    replaces them with what the episodes prove.
    """
    return RunState(
        run_id=state.run_id,
        goal=state.goal,
        status=state.status,
        history=state.history,
        best=recompute_best(state.history, state.goal.objective),
        budget=recompute_budget(state.budget, state.history),
        step_idx=recompute_step_idx(state.history),
        seed=state.seed,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


def goal_satisfied(best: Best | None, goal: Goal) -> bool:
    """The SUCCESS exit: good enough, stop early.

    Inactive in Phase 1 because the benchmark goals set no target — the runs end
    on the evaluation budget. The check is here because it is deterministic and
    belongs to the loop's structure, not to the agent.
    """
    if best is None or goal.objective.target is None:
        return False
    if not best.result.feasible:
        return False
    if goal.objective.direction.value == "MINIMISE":
        return best.result.objective_value <= goal.objective.target
    return best.result.objective_value >= goal.objective.target


class RunRepositoryProtocol(Protocol):
    """The slice of the repository the loop actually needs.

    Declared here rather than importing ``RunRepository`` so that the loop has
    no dependency on Postgres, SQLAlchemy or any of it. Tests substitute an
    in-memory fake and the loop cannot tell the difference.
    """

    def save_run(self, state: RunState, *, simulator: str, strategy: str) -> None: ...

    def append_episode(self, run_id: UUID, episode: Episode) -> bool: ...


@dataclass(frozen=True)
class LoopResult:
    state: RunState
    reason: TerminationReason
    episodes_run: int


# Called after each episode is durably written. Used by the CLI to print
# progress; keeping it a callback keeps printing out of the loop's logic.
EpisodeCallback = Callable[[RunState, Episode], None]


def run_loop(
    *,
    state: RunState,
    simulator: Simulator,
    strategy: Strategy,
    repo: RunRepositoryProtocol,
    on_episode: EpisodeCallback | None = None,
    crash_at: int | None = None,
) -> LoopResult:
    """Run episodes until a termination condition fires.

    ``state`` may be a fresh run or one loaded from the database; the loop does
    not need to know which, because it recomputes its position either way.

    ``crash_at`` is a development flag. When set, the process is killed with
    ``os._exit`` immediately before the episode with that index is written —
    the worst possible moment, after the simulation has been paid for but
    before anything records it. It exists so the durability claim can be tested
    automatically rather than only by hand.
    """
    state = rehydrate(state)
    goal = state.goal
    episodes_run = 0

    while True:
        budget = state.budget

        # Terminate on evaluations spent, NOT on episodes run.
        #
        # Phase 1 checked `state.step_idx >= budget.max_evaluations` as well.
        # The two agree exactly as long as every episode reaches the simulator,
        # which is true for a sampler that cannot propose anything invalid — so
        # the disagreement was invisible. From Phase 3 a rejected proposal
        # advances step_idx without spending an evaluation, and the old check
        # would have quietly given a hallucinating planner fewer simulator
        # calls than a well-behaved one. The evaluation budget is the thing
        # that has to be equal across strategies for the ablation to mean
        # anything, so it is the thing the loop counts.
        if budget.exhausted():
            reason = TerminationReason.BUDGET
            break
        if budget.rejections_exceeded():
            reason = TerminationReason.REJECTIONS
            break
        if goal_satisfied(state.best, goal):
            reason = TerminationReason.SUCCESS
            break

        idx = state.step_idx
        started = time.perf_counter()

        try:
            candidate = strategy.propose(goal, state.history, episode_rng(state.seed, idx))
        except StrategyExhausted:
            # An enumerating strategy has covered its space. Not an error, and
            # not the same thing as running out of budget.
            reason = TerminationReason.EXHAUSTED
            break

        # The guard runs before the simulator, always. In Phase 1 the proposer
        # is a uniform sampler that cannot produce an invalid candidate; from
        # Phase 3 it is an LLM that can, and this line is unchanged.
        outcome: SimResult | ToolError
        try:
            guarded = validate(candidate, goal.parameter_space)
            candidate = guarded.candidate
            outcome = simulator.run(candidate.params)
        except GuardRejection as exc:
            outcome = ToolError(kind="GuardRejection", message=exc.reason)
        except Exception as exc:  # a broken simulator must not lose the run
            outcome = ToolError(kind=type(exc).__name__, message=str(exc), retryable=True)

        # improved / delta / feasible are COMPUTED here, from oracle output.
        # From Phase 4 they are passed into the critic, which explains them but
        # cannot change them. This is Rule 2.
        if isinstance(outcome, SimResult):
            improved = outcome.better_than(state.best, goal.objective)
            delta = outcome.delta_vs(state.best, goal.objective)
            feasible = outcome.feasible
        else:
            improved, delta, feasible = False, 0.0, False

        episode = Episode(
            idx=idx,
            candidate=candidate,
            result=outcome,
            evaluation=Evaluation.computed_only(
                improved=improved, delta_vs_best=delta, feasible=feasible
            ),
            decision=ReplanDecision.placeholder(),
            cost=CostRecord.zero(),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

        if crash_at is not None and idx == crash_at:
            # os._exit skips cleanup handlers, flushes nothing and gives no
            # chance to tidy up — which is the point. It is as close to
            # `kill -9` as a process can do to itself.
            print(f"[crash-at] killing process before writing episode {idx}", flush=True)
            os._exit(1)

        # ---- commit point -------------------------------------------------
        inserted = repo.append_episode(state.run_id, episode)
        if not inserted:
            print(
                f"episode {idx} of run {state.run_id} already exists - another process is "
                "writing to this run. Stopping. (Run locking arrives in Phase 3.)",
                file=sys.stderr,
            )
            reason = TerminationReason.ERROR
            break
        # -------------------------------------------------------------------

        best = state.best
        if isinstance(outcome, SimResult):
            # Only a real simulator call spends the evaluation budget.
            next_budget = budget.with_evaluation(wall_time_s=outcome.wall_time_s, cost=episode.cost)
            if improved:
                best = Best(episode_idx=idx, candidate=candidate, result=outcome)
        else:
            next_budget = budget.with_rejection(cost=episode.cost)

        state = state.advanced(
            episode=episode,
            best=best,
            budget=next_budget,
            status=RunStatus.EXECUTING,
        )
        repo.save_run(state, simulator=simulator.name, strategy=strategy.name)

        episodes_run += 1
        if on_episode is not None:
            on_episode(state, episode)

    final_status = RunStatus.FAILED if reason is TerminationReason.ERROR else RunStatus.DONE
    state = state.with_status(final_status)
    repo.save_run(state, simulator=simulator.name, strategy=strategy.name)

    return LoopResult(state=state, reason=reason, episodes_run=episodes_run)
