"""The loop.

The structure is TECHNICAL_DESIGN §3, complete from Phase 4: a strategy
proposes, the guard validates, the simulator decides, and — for a strategy that
implements ``Reflects`` — a critic explains the result and a replanner chooses
what to do next. A strategy that does not implement it stores the computed
evaluation and a placeholder decision, exactly as every strategy did through
Phase 3.5. The loop never learns what a critic is; it asks
``isinstance(strategy, Reflects)`` and nothing more.

**Where Rule 2 lives in this file.** ``improved``, ``delta_vs_best`` and
``feasible`` are computed here from oracle output and passed *into* the critic.
The critic's schema does not contain them, and the stored ``Evaluation`` is
assembled by ``evaluation_from`` out of the oracle's numbers and the model's
words. There is no path by which a model's opinion reaches a computed field, and
that is a property of the types rather than of the prompt.

Everything else interesting in this file is about one property — **durable
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sil_agent.agent.critic import CRITIC_UNAVAILABLE, evaluation_from
from sil_agent.agent.guards import GuardRejection, is_duplicate, perturb, validate
from sil_agent.agent.stagnation import StagnationDetector, check_all
from sil_agent.agent.state import (
    Best,
    BudgetState,
    Candidate,
    CandidateSource,
    CostRecord,
    Episode,
    Evaluation,
    Goal,
    Objective,
    ReplanAction,
    ReplanDecision,
    RunState,
    RunStatus,
    SimResult,
    TerminationReason,
    ToolError,
)
from sil_agent.services.retry import ProviderError
from sil_agent.simulators.base import Simulator
from sil_agent.strategies.base import Reflects, ReportsCost, Strategy, StrategyExhausted


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


def perturb_rng(seed: int, idx: int) -> random.Random:
    """A separate deterministic stream for perturbation.

    Distinct from ``episode_rng`` so that nudging a duplicate cannot consume
    draws a strategy was going to use, which would make perturbation change the
    *next* proposal as a side effect. Both are derived from persisted values, so
    Rule 1 holds either way.
    """
    return random.Random(f"{seed}:{idx}:perturb")


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
    # Which detector fired, when the reason is STAGNATION. Empty otherwise.
    detail: str = ""


# Called after each episode is durably written. Used by the CLI to print
# progress; keeping it a callback keeps printing out of the loop's logic.
EpisodeCallback = Callable[[RunState, Episode], None]

# Called after each episode is durably written, to say whether this worker still
# owns the run. The harness passes ``RunLock.renew``. Returning False means the
# lease was lost and another worker may already be writing, so this one must
# stop immediately — see ``eval/harness.py``.
Heartbeat = Callable[[], bool]


def run_loop(
    *,
    state: RunState,
    simulator: Simulator,
    strategy: Strategy,
    repo: RunRepositoryProtocol,
    on_episode: EpisodeCallback | None = None,
    crash_at: int | None = None,
    detectors: Sequence[StagnationDetector] | None = None,
    heartbeat: Heartbeat | None = None,
    honour_terminate: bool = False,
) -> LoopResult:
    """Run episodes until a termination condition fires.

    ``state`` may be a fresh run or one loaded from the database; the loop does
    not need to know which, because it recomputes its position either way.

    ``crash_at`` is a development flag. When set, the process is killed with
    ``os._exit`` immediately before the episode with that index is written —
    the worst possible moment, after the simulation has been paid for but
    before anything records it. It exists so the durability claim can be tested
    automatically rather than only by hand.

    ``honour_terminate`` decides whether a replanner recommending ``TERMINATE``
    actually stops the run. **It defaults to off, and the ablation runs with it
    off.** Two reasons, and the second is the operative one:

    * Rule 2 — termination is a decision, and deterministic code decides.
    * Fairness — the evaluation budget has to buy the same number of simulator
      calls for every strategy (``TECHNICAL_DESIGN.md`` §6). A strategy that
      talks itself into quitting at evaluation six has not lost the same contest
      the others were in, it has set its own budget.

    The recommendation is recorded on the episode either way, and the report
    prints the rate. A number is strictly more informative than a confound.
    """
    state = rehydrate(state)
    goal = state.goal
    episodes_run = 0
    stagnation_detail = ""

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

        # The third independent exit (TECHNICAL_DESIGN §3). Off by default so
        # the Phase 2 baselines are unaffected: random search legitimately goes
        # long stretches without improving, and terminating it early would
        # change the numbers the whole comparison rests on.
        if detectors:
            verdict = check_all(detectors, state.history, goal)
            if verdict.stuck:
                stagnation_detail = verdict.detail
                reason = TerminationReason.STAGNATION
                break

        idx = state.step_idx
        started = time.perf_counter()

        outcome: SimResult | ToolError
        proposal_failed: ToolError | None = None

        try:
            candidate = strategy.propose(goal, state.history, episode_rng(state.seed, idx))
        except StrategyExhausted:
            # An enumerating strategy has covered its space. Not an error, and
            # not the same thing as running out of budget.
            reason = TerminationReason.EXHAUSTED
            break
        except ProviderError as exc:
            # From Phase 3 the proposer is an LLM, and it can fail to produce
            # anything usable: unparseable output, a schema it will not satisfy,
            # or every provider exhausted. That is an ordinary event, not a
            # crash. It is recorded as a failed episode so the history shows what
            # happened, and it counts against the rejection allowance so a
            # permanently broken provider ends the run instead of looping.
            #
            # A placeholder candidate is stored because an Episode needs one and
            # there genuinely is not a proposal. Empty params say exactly that.
            candidate = Candidate(
                params={},
                rationale="the planner produced no usable proposal",
                source=CandidateSource.PLANNER,
            )
            proposal_failed = ToolError(
                kind=type(exc).__name__, message=str(exc), retryable=True
            )

        # The guard runs before the simulator, always. In Phase 1 the proposer
        # was a uniform sampler that could not produce an invalid candidate;
        # from Phase 3 it is an LLM that can, and these lines are unchanged.
        if proposal_failed is not None:
            outcome = proposal_failed
        else:
            try:
                guarded = validate(candidate, goal.parameter_space)
                candidate = guarded.candidate

                # TECHNICAL_DESIGN §3. A planner shown a history with one good
                # point will propose that same point again, with a fluent
                # justification each time, and spend a whole budget re-measuring
                # it. Perturbation is the deterministic answer to a model that
                # will not explore. The episode is marked PERTURB so the rate is
                # visible in the report rather than hidden.
                if is_duplicate(candidate, state.history):
                    candidate = perturb(
                        candidate,
                        goal.parameter_space,
                        perturb_rng(state.seed, idx),
                    )

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

        # What the proposal cost. Zero for the baselines, which do not implement
        # the optional protocol; real tokens for an LLM strategy. Read after
        # propose() so a rejected or failed proposal is still charged — the call
        # was paid for whether or not it produced anything usable.
        cost = strategy.last_cost if isinstance(strategy, ReportsCost) else CostRecord.zero()

        # The computed verdict. These three numbers are the oracle's, and from
        # here they are passed *into* the critic rather than asked of it.
        computed = Evaluation.computed_only(
            improved=improved, delta_vs_best=delta, feasible=feasible
        )
        evaluation = computed
        decision = ReplanDecision.placeholder()

        # ---- reflection (Phase 4) -----------------------------------------
        #
        # Optional, exactly like ReportsCost above: a strategy that does not
        # implement the protocol stores the computed evaluation and a
        # placeholder decision, which is what every strategy did through Phase
        # 3.5. The loop never learns what a critic is.
        if isinstance(strategy, Reflects):
            try:
                reflection = strategy.reflect(
                    goal, state.history, candidate, outcome, computed, state.best
                )
            except Exception as exc:
                # A backstop, not the ordinary path — a well-behaved reflector
                # reports model failures through `Reflection.failure` and keeps
                # its partial cost. Either way the simulation has already been
                # paid for, and losing the episode to a failure in the narration
                # would be the most expensive possible response to it.
                evaluation = Evaluation(
                    improved=improved,
                    delta_vs_best=delta,
                    feasible=feasible,
                    diagnosis=f"{CRITIC_UNAVAILABLE}: {type(exc).__name__}: {exc}",
                )
            else:
                # The one place the two halves are joined. `evaluation_from`
                # takes the computed fields from `computed` and only the prose
                # from the verdict, so a reflector cannot revise a grade.
                evaluation = evaluation_from(computed, reflection.verdict)
                decision = reflection.decision
                cost = cost.plus(reflection.cost)
        # -------------------------------------------------------------------

        episode = Episode(
            idx=idx,
            candidate=candidate,
            result=outcome,
            evaluation=evaluation,
            decision=decision,
            cost=cost,
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
            # Phase 4 wired the Redis lock into the harness, so two workers
            # should no longer reach this line. It stays, and stays
            # load-bearing: a lock has a TTL and a stalled worker can outlive
            # its lease, whereas the natural key on (run_id, idx) cannot be
            # outlived. The lock removes the waste; this removes the corruption.
            print(
                f"episode {idx} of run {state.run_id} already exists - another process is "
                "writing to this run. Stopping.",
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

        # Extend the lease, now that an episode is durably written. Placed after
        # the commit rather than before it so a lost lock is discovered with the
        # work safely recorded — and placed at all because an agent_full episode
        # is three model calls, which is long enough for a fixed TTL to be the
        # wrong tool.
        if heartbeat is not None and not heartbeat():
            print(
                f"run {state.run_id}: lost the run lock after episode {idx}. Another "
                "worker may now own this run, so this one stops writing.",
                file=sys.stderr,
            )
            reason = TerminationReason.ERROR
            break

        # The replanner asked to stop. Off by default — see the docstring.
        if honour_terminate and decision.action is ReplanAction.TERMINATE:
            reason = TerminationReason.STAGNATION
            stagnation_detail = f"replanner recommended TERMINATE: {decision.reason}"
            break

    final_status = RunStatus.FAILED if reason is TerminationReason.ERROR else RunStatus.DONE
    state = state.with_status(final_status)
    repo.save_run(state, simulator=simulator.name, strategy=strategy.name)

    return LoopResult(
        state=state,
        reason=reason,
        episodes_run=episodes_run,
        detail=stagnation_detail,
    )
