"""The LLM strategies: one that loops, and one that does not.

Both satisfy the same ``Strategy`` protocol as random search, so the Phase 2
harness runs them without modification and they appear in the same report. That
was the point of building the harness first.

``AgentNoReflection`` is the Phase 3 deliverable: a planner that sees history and
the incumbent best, and proposes the next point. No critic — that is Phase 4, and
keeping it out is what makes Phase 4 a clean experiment rather than a comparison
against a moving target.

``SingleShotLLM`` is the control that makes the loop's value measurable. It asks
once for the entire batch of points, evaluates them all, and never sees a result.
If the looping agent cannot beat it, the loop is not earning its cost, and
"the agent beat random search" would be indistinguishable from "the model has
read about Branin".

Both are honest about what they cannot promise. Phases 1 and 2 guaranteed that a
resumed run reproduces an uninterrupted one byte for byte. An LLM cannot promise
that even at temperature zero, because providers batch requests and route across
hardware that reorders floating-point work. What survives is the *structural*
guarantee — resume continues at the correct episode, loses nothing and repeats
nothing — plus replay from the recorded calls, which recovers exactness for any
run that has already happened. See ``docs/phases/phase-03-brief.md``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from sil_agent.agent.planner import (
    BatchProposal,
    Planner,
    describe_constraints,
    describe_objective,
    describe_space,
)
from sil_agent.agent.state import (
    Best,
    Candidate,
    CandidateSource,
    CostRecord,
    Direction,
    Episode,
    Goal,
    SimResult,
)
from sil_agent.prompts import load
from sil_agent.services.router import ModelRouter, Prompt, Role
from sil_agent.strategies.base import StrategyExhausted


def recompute_best(history: Sequence[Episode], goal: Goal) -> Best | None:
    """The incumbent, replayed from history.

    Duplicated in spirit from ``agent/loop.py`` because ``Strategy.propose``
    receives history but not ``best`` — the protocol was fixed in Phase 1 and
    changing it would touch every strategy. Recomputing is cheap and keeps the
    planner a pure function of what is persisted.
    """
    best: Best | None = None
    for episode in history:
        result = episode.sim_result
        if result is None:
            continue
        if result.better_than(best, goal.objective):
            best = Best(episode_idx=episode.idx, candidate=episode.candidate, result=result)
    return best


class AgentNoReflection:
    """Planner only. Sees history and best; no critic, no memory across runs.

    ``last_cost`` is how the LLM spend for an episode reaches the loop. It is
    genuinely transient — written on every call, read immediately after, and
    never used to decide anything — so it does not violate Rule 1. The decision
    the loop makes is still determined entirely by persisted state.
    """

    def __init__(
        self,
        router: ModelRouter,
        *,
        max_evaluations: int,
        prompt_version: str = "v1",
    ) -> None:
        self._planner = Planner(router, prompt_version=prompt_version)
        self._max_evaluations = max_evaluations
        self.last_cost: CostRecord = CostRecord.zero()

    @property
    def name(self) -> str:
        return "agent_no_reflection"

    def propose(
        self,
        goal: Goal,
        history: list[Episode],
        rng: random.Random,
    ) -> Candidate:
        """``rng`` is unused: the randomness lives in the model, not here."""
        best = recompute_best(history, goal)
        candidate, cost = self._planner.propose(
            goal, history, best, max_evaluations=self._max_evaluations
        )
        self.last_cost = cost
        return candidate


class SingleShotLLM:
    """Asks once for every point, then serves them one at a time.

    The batch is regenerated on every call rather than stored on the object.
    That looks wasteful and is the opposite: with a ``CachingRouter`` the second
    and later calls hit the recorded reply, so the run makes exactly one real
    provider call while remaining a pure function of persisted state. Holding the
    batch in memory would be one line shorter and would break resume — the
    restarted process would ask the model again and get a different plan.
    """

    def __init__(
        self,
        router: ModelRouter,
        *,
        max_evaluations: int,
        prompt_version: str = "v1",
    ) -> None:
        self._router = router
        self._max_evaluations = max_evaluations
        self._template = load("single_shot", prompt_version)
        self.last_cost: CostRecord = CostRecord.zero()

    @property
    def name(self) -> str:
        return "single_shot_llm"

    def propose(
        self,
        goal: Goal,
        history: list[Episode],
        rng: random.Random,
    ) -> Candidate:
        index = max((episode.idx for episode in history), default=-1) + 1

        system, user = self._template.render(
            objective=describe_objective(goal),
            parameter_space=describe_space(goal.parameter_space),
            constraints=describe_constraints(goal),
            count=self._max_evaluations,
        )

        batch, cost = self._router.complete(
            Role.PLANNER,
            Prompt(
                system=system,
                user=user,
                template_version=self._template.identifier,
                # The whole plan arrives in one reply, so the ceiling scales
                # with the budget — and has to cover two different things.
                #
                # Reasoning tokens count against the same ceiling as visible
                # output on a thinking model, and on `gemini-flash-latest` they
                # cannot be turned off (`thinkingConfig.thinkingBudget: 0` is
                # rejected with a 400). Measured, thinking costs roughly 2,000
                # tokens flat whatever the request, while the visible JSON runs
                # about 55 tokens per proposal.
                #
                # Reasoning does not scale with the *reply* — it scales with how
                # hard the model finds the question. The same 50-proposal
                # request needed 1,700 reasoning tokens with a terse prompt and
                # blew past 11,500 with this strategy's full instructions.
                # Budget for the pessimistic case rather than the measured one.
                #
                # Getting this wrong is expensive to diagnose: an under-budgeted
                # call emitted 198 characters and reported MAX_TOKENS, which
                # looks nothing like "the plan did not fit". Only generated
                # tokens are billed, so the ceiling itself costs nothing.
                max_tokens=max(16_000, 8_000 + 300 * self._max_evaluations),
            ),
            BatchProposal,
        )
        self.last_cost = cost

        if index >= len(batch.proposals):
            # The model returned fewer points than asked for. Not an error worth
            # failing the run over: the strategy has genuinely run out of plan,
            # which is exactly what EXHAUSTED means. The report shows the run
            # used fewer evaluations than its budget, which is the honest record
            # of what the model did.
            raise StrategyExhausted(
                f"single-shot plan contained {len(batch.proposals)} proposals, "
                f"fewer than the {self._max_evaluations} requested"
            )

        proposal = batch.proposals[index]
        return Candidate(
            params=proposal.params,
            rationale=proposal.rationale,
            source=CandidateSource.PLANNER,
        )


def objective_of(result: SimResult, direction: Direction) -> float:
    """Signed objective, so callers can always minimise."""
    return result.objective_value if direction is Direction.MINIMISE else -result.objective_value
