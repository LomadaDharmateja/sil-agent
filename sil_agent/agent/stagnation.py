"""Detecting that the agent has stopped making progress.

``TECHNICAL_DESIGN.md`` §3 lists this as the third independent exit, alongside
success and budget exhaustion. It is worth having for a reason the design states
plainly: toy agents run forever or stop after a fixed N, and noticing your own
lack of progress is rare.

It also has a concrete purpose here. From Phase 3 an episode costs a model call,
so twenty episodes spent re-proposing the same point are twenty calls of quota
spent learning nothing. Stopping early leaves that budget for a run that is
still improving.

**Detectors are a list, not a hard-coded chain.** Each one answers "is this run
stuck?" independently and any single fire terminates. Two are implemented now;
the other two in the design — critic confidence declining, and repeated
near-identical diagnoses — need a critic to exist, so their interface is settled
here and they arrive in Phase 4.

**Stagnation is not failure.** A run that stops at episode 30 of 50 has not spent
its budget, exactly like grid search exhausting its grid, and the report must say
so rather than showing a strategy that scored badly on a contest it left early.
``TerminationReason.STAGNATION`` is already distinct from BUDGET.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sil_agent.agent.state import Episode, Goal, ParamKind


@dataclass(frozen=True)
class StagnationVerdict:
    """Whether a run is stuck, and the reason to put in the record."""

    stuck: bool
    detector: str = ""
    detail: str = ""

    @classmethod
    def moving(cls) -> StagnationVerdict:
        return cls(stuck=False)


class StagnationDetector(Protocol):
    """One reason to believe a run has stopped making progress."""

    @property
    def name(self) -> str: ...

    def check(self, history: Sequence[Episode], goal: Goal) -> StagnationVerdict: ...


class NoImprovement:
    """No new best in the last N evaluations.

    The obvious detector, and the one that catches most real stalls. It counts
    *evaluations*, not episodes: a run whose last eight proposals were all
    rejected by the guard has not had eight chances to improve, and terminating
    it as stagnant would blame the search for what is a proposal-quality problem
    — which the separate rejection allowance already handles.
    """

    def __init__(self, window: int = 8) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self._window = window

    @property
    def name(self) -> str:
        return "no_improvement"

    def check(self, history: Sequence[Episode], goal: Goal) -> StagnationVerdict:
        evaluated = [e for e in history if e.sim_result is not None]
        if len(evaluated) < self._window:
            return StagnationVerdict.moving()

        recent = evaluated[-self._window :]
        if any(episode.evaluation.improved for episode in recent):
            return StagnationVerdict.moving()

        return StagnationVerdict(
            stuck=True,
            detector=self.name,
            detail=f"no improvement in the last {self._window} evaluations",
        )


class DiversityCollapse:
    """Recent proposals are all crowding the same point.

    Measured in *normalised* space — each parameter scaled to [0, 1] by its
    declared bounds — because raw distances are meaningless when one parameter
    ranges over [0, 1] and another over [-5, 10]. Without normalisation this
    detector would be dominated by whichever parameter happens to have the
    widest range.

    Distinct from ``NoImprovement``: an agent can keep finding tiny improvements
    while circling one point, which is exploitation that has run its course.
    """

    def __init__(self, window: int = 6, epsilon: float = 0.02) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 to compare points")
        self._window = window
        self._epsilon = epsilon

    @property
    def name(self) -> str:
        return "diversity_collapse"

    def check(self, history: Sequence[Episode], goal: Goal) -> StagnationVerdict:
        evaluated = [e for e in history if e.sim_result is not None]
        if len(evaluated) < self._window:
            return StagnationVerdict.moving()

        points = [
            normalise(episode.candidate.params, goal) for episode in evaluated[-self._window :]
        ]
        points = [p for p in points if p]
        if len(points) < 2:
            return StagnationVerdict.moving()

        spread = max_pairwise_distance(points)
        if spread > self._epsilon:
            return StagnationVerdict.moving()

        return StagnationVerdict(
            stuck=True,
            detector=self.name,
            detail=(
                f"the last {self._window} proposals span {spread:.4f} in normalised space, "
                f"within the {self._epsilon} collapse threshold"
            ),
        )


def normalise(params: dict[str, float | int | str], goal: Goal) -> list[float]:
    """Scale numeric parameters to [0, 1] using their declared bounds.

    Categorical parameters are skipped: there is no meaningful distance between
    two names, and inventing one (index order, say) would make the detector
    sensitive to how the choices happen to be listed.
    """
    coordinates: list[float] = []
    for spec in goal.parameter_space.params:
        if spec.kind is ParamKind.CATEGORICAL or spec.bounds is None:
            continue
        value = params.get(spec.name)
        if not isinstance(value, int | float):
            continue
        low, high = spec.bounds
        span = high - low
        coordinates.append((float(value) - low) / span if span else 0.0)
    return coordinates


def max_pairwise_distance(points: Sequence[Sequence[float]]) -> float:
    """The widest gap between any two points — the diameter of the cloud.

    Diameter rather than mean distance: an agent that proposes five identical
    points and one distant one is still exploring, and a mean would hide that
    behind the four duplicates.
    """
    widest = 0.0
    for i, first in enumerate(points):
        for second in points[i + 1 :]:
            distance = sum((a - b) ** 2 for a, b in zip(first, second, strict=False)) ** 0.5
            widest = max(widest, distance)
    return widest


def default_detectors() -> list[StagnationDetector]:
    """The detectors a run uses unless told otherwise.

    Windows are deliberately generous. A detector that fires early costs a real
    result; one that fires late costs some quota. On a 50-evaluation budget,
    eight flat evaluations is a genuine stall rather than bad luck.
    """
    return [NoImprovement(window=8), DiversityCollapse(window=6, epsilon=0.02)]


def check_all(
    detectors: Sequence[StagnationDetector],
    history: Sequence[Episode],
    goal: Goal,
) -> StagnationVerdict:
    """First detector to fire wins. Returns a moving verdict if none do."""
    for detector in detectors:
        verdict = detector.check(history, goal)
        if verdict.stuck:
            return verdict
    return StagnationVerdict.moving()
