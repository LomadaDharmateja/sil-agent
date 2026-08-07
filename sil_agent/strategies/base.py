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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sil_agent.agent.critic import CriticVerdict
from sil_agent.agent.state import (
    Best,
    Candidate,
    CostRecord,
    Episode,
    Evaluation,
    Goal,
    ReplanDecision,
    SimResult,
    ToolError,
)


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


@dataclass(frozen=True)
class Reflection:
    """What a reflecting strategy produces after seeing one result.

    ``verdict`` carries only the model's prose. The computed fields —
    ``improved``, ``delta_vs_best``, ``feasible`` — are absent from
    :class:`~sil_agent.agent.critic.CriticVerdict` by design, and the loop
    assembles the stored ``Evaluation`` from the oracle's numbers and this
    verdict's words. A reflector has no channel through which to change a grade.
    """

    verdict: CriticVerdict
    decision: ReplanDecision
    cost: CostRecord
    # Set when reflection did not complete — the critic or the replanner failed.
    # Recorded rather than raised: the simulator call has already been paid for
    # and the episode must still be written. Counted in the report, so a run
    # whose critic was down is distinguishable from one that never had a critic.
    failure: str | None = None


@runtime_checkable
class Reflects(Protocol):
    """A strategy that diagnoses its own results and re-plans from the diagnosis.

    Optional and separate from ``Strategy`` for exactly the reason
    ``ReportsCost`` is: random search does not reflect and should not have to
    implement a method saying so. The loop checks for this at runtime and stores
    the computed-only evaluation plus a placeholder decision when it is absent —
    which is precisely what Phases 1 to 3.5 did for every strategy.

    Keeping it here rather than adding a flag to the harness means
    ``build_strategy(name)`` stays the single source of truth for what a
    strategy is. A second switch elsewhere is how a run ends up labelled
    ``agent_full`` while quietly not reflecting.
    """

    def reflect(
        self,
        goal: Goal,
        history: Sequence[Episode],
        candidate: Candidate,
        outcome: SimResult | ToolError,
        computed: Evaluation,
        best: Best | None,
    ) -> Reflection:
        """Diagnose ``outcome`` and choose what to do next.

        ``computed`` is the oracle's verdict, passed *in*: the reflector is told
        whether the result improved and by how much, and its prompt renders
        those as facts to be explained. ``history`` is every episode before this
        one, and ``best`` the incumbent it was compared against.

        Must not raise for an ordinary model failure — return a ``Reflection``
        with ``failure`` set instead, so the episode is still written.
        """
        ...
