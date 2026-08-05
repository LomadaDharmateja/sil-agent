"""The strategy interface — the one thing the ablation swaps.

Every row of the Phase 2 comparison table (random, grid, TPE, single-shot LLM,
agent without reflection, full agent) is an implementation of this protocol.
Same simulator, same budget, same harness; only this changes. That is what makes
the comparison fair.

One deviation from TECHNICAL_DESIGN §9, which has
``propose(goal, history) -> Candidate``: the random number generator is passed in
explicitly rather than held on the strategy object.

The reason is Rule 1. A strategy holding its own RNG is hidden in-memory state:
resume a run in a fresh process and that generator restarts from the beginning,
so episode 11 after a restart draws what episode 0 drew. Passing an RNG derived
from ``(seed, episode_idx)`` — both of which are persisted — keeps the loop a
pure function of stored state, and is what makes "kill it and resume, get the
identical sequence" true rather than nearly true.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from sil_agent.agent.state import Candidate, CostRecord, Episode, Goal


class StrategyExhausted(Exception):
    """The strategy has nothing left to propose, with budget still remaining.

    Only an *enumerating* strategy can be exhausted. Grid search over a
    6-dimensional space fits 2 points per dimension into a 200-evaluation
    budget, covers all 64 of them, and is then genuinely finished — asking it
    for a 65th point has no answer.

    Raised rather than returning ``None`` so that the ordinary path keeps a
    single, non-optional return type, and so a strategy cannot silently forget
    to signal it. The loop catches this and terminates with
    ``TerminationReason.EXHAUSTED``, which is reported distinctly from BUDGET:
    "used 64 of 200 evaluations because it ran out of grid" and "used all 200"
    are different facts about a strategy and the comparison table has to say
    which happened.
    """


class Strategy(Protocol):
    """Proposes the next candidate to evaluate."""

    @property
    def name(self) -> str:
        """Stable identifier, recorded with the run so results are attributable."""
        ...

    def propose(
        self,
        goal: Goal,
        history: list[Episode],
        rng: random.Random,
    ) -> Candidate:
        """Propose one candidate.

        ``history`` is every completed episode, oldest first. ``rng`` is seeded
        deterministically from the run's seed and the episode index.

        Raises:
            StrategyExhausted: if there is nothing left to propose.
        """
        ...


@runtime_checkable
class ReportsCost(Protocol):
    """A strategy that spent money proposing, and can say how much.

    Optional, and separate from ``Strategy`` on purpose: the baselines cost
    nothing and should not have to implement an accounting method to say so.
    The loop checks for this at runtime and records zero when it is absent.

    ``last_cost`` describes the call that has just happened. It is written and
    read within a single episode and never influences a decision, so it is not
    the hidden state Rule 1 forbids — the next episode is still determined
    entirely by what is in the database.
    """

    @property
    def last_cost(self) -> CostRecord: ...
