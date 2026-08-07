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


class ConfidenceDecline:
    """The critic's own confidence is falling across a window.

    The third detector in ``TECHNICAL_DESIGN.md`` §3, and one that could not
    exist before Phase 4 because there was no critic to be confident. It is the
    agent saying, in a structured field rather than in prose, that it is running
    out of explanations.

    **Halves rather than strict monotonicity.** "Declining" implemented as
    ``c[0] > c[1] > c[2] > ...`` never fires on real output: a model's
    self-reported confidence is noisy at the second decimal and a single 0.4
    between two 0.3s resets it forever. Comparing the mean of the first half of
    the window against the mean of the second half is robust to that and still
    means what the design says.

    Only episodes that actually carry a diagnosis are counted. An episode
    reflected on by a failed critic, or produced by a strategy with no critic at
    all, has ``confidence`` at its default 0.0, and letting those into the
    window would make "the critic was down" indistinguishable from "the critic
    has given up".
    """

    def __init__(self, window: int = 6, min_drop: float = 0.15) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 to compare halves")
        self._window = window
        self._min_drop = min_drop

    @property
    def name(self) -> str:
        return "confidence_decline"

    def check(self, history: Sequence[Episode], goal: Goal) -> StagnationVerdict:
        scored = [e for e in history if e.evaluation.diagnosis]
        if len(scored) < self._window:
            return StagnationVerdict.moving()

        recent = scored[-self._window :]
        confidences = [e.evaluation.confidence for e in recent]

        half = len(confidences) // 2
        earlier = confidences[:half]
        later = confidences[len(confidences) - half :]

        earlier_mean = sum(earlier) / len(earlier)
        later_mean = sum(later) / len(later)
        drop = earlier_mean - later_mean

        if drop < self._min_drop:
            return StagnationVerdict.moving()

        return StagnationVerdict(
            stuck=True,
            detector=self.name,
            detail=(
                f"critic confidence fell from {earlier_mean:.2f} to {later_mean:.2f} "
                f"across the last {self._window} diagnoses "
                f"(drop {drop:.2f} exceeds {self._min_drop})"
            ),
        )


class RepeatedDiagnosis:
    """The critic has been saying the same thing for several episodes.

    The fourth detector in ``TECHNICAL_DESIGN.md`` §3, and the instrument for
    the failure mode the Phase 4 brief expects most: a small model that produces
    fluent, plausible, *identical* analysis of every result it is shown. That is
    not an error and nothing else in the system notices it — the episodes look
    healthy, the JSON validates, and the loop learns nothing.

    **Token overlap, not string equality.** A model restates the same thought in
    different words every time, so ``==`` never fires. Jaccard similarity over
    the word sets — the size of the intersection divided by the size of the
    union — catches "this region appears unpromising" against "the region seems
    unpromising here" while separating both from a genuinely different analysis.

    **The minimum pairwise similarity, not the mean.** The claim is that the
    last N diagnoses are *all* near-identical. A mean would let one repeated
    pair carry a window that also contains a real change of mind.
    """

    def __init__(self, window: int = 4, threshold: float = 0.75) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 to compare diagnoses")
        self._window = window
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "repeated_diagnosis"

    def check(self, history: Sequence[Episode], goal: Goal) -> StagnationVerdict:
        diagnoses = [e.evaluation.diagnosis for e in history if e.evaluation.diagnosis]
        if len(diagnoses) < self._window:
            return StagnationVerdict.moving()

        recent = [tokenise(text) for text in diagnoses[-self._window :]]
        if any(not tokens for tokens in recent):
            # An empty token set after filtering — a diagnosis of "ok", say.
            # Similarity is undefined rather than 1.0, and treating it as
            # identical would fire the detector on a critic that said nothing.
            return StagnationVerdict.moving()

        weakest = min(
            jaccard(first, second)
            for index, first in enumerate(recent)
            for second in recent[index + 1 :]
        )

        if weakest < self._threshold:
            return StagnationVerdict.moving()

        return StagnationVerdict(
            stuck=True,
            detector=self.name,
            detail=(
                f"the last {self._window} diagnoses overlap by at least "
                f"{weakest:.0%}, above the {self._threshold:.0%} repetition threshold"
            ),
        )


# Words carrying no diagnostic content. Kept short deliberately: a long stop
# list starts deciding what counts as a repeated thought, which is the
# detector's job and not a constant's.
_STOPWORDS = frozenset(
    {
        "the", "and", "that", "this", "with", "for", "from", "was", "were", "has",
        "have", "had", "not", "but", "which", "into", "than", "then", "its",
        "are", "been", "being", "would", "could", "should", "may", "might",
    }
)


def stem(word: str) -> str:
    """Strip the commonest English inflections. Crude on purpose.

    Added after the first version of ``RepeatedDiagnosis`` failed its own test
    case: two diagnoses differing only in "suggesting" against "suggests" scored
    0.71 and slipped under a 0.75 threshold. Inflection *is* the same thought in
    different words, which is precisely what the detector claims to see through,
    so treating those as distinct tokens made it measure grammar rather than
    content.

    Four suffixes and a length floor, not a real stemmer. A proper one is a
    dependency and a vocabulary, and everything past this point is diminishing:
    the threshold above already requires three quarters of the content words to
    match, so the occasional over-merge cannot carry a window on its own.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenise(text: str) -> frozenset[str]:
    """Lowercase stemmed word set, minus stopwords, short tokens and bare numbers.

    Numbers are dropped because two diagnoses that differ only in the values
    they quote — "fell from 6.1 to 5.8" against "fell from 5.8 to 5.5" — are the
    same thought applied twice, which is exactly what this detector is looking
    for. Keeping them would make boilerplate look novel every episode.
    """
    lowered = "".join(char if char.isalnum() else " " for char in text.lower())
    return frozenset(
        stem(word)
        for word in lowered.split()
        if len(word) >= 3 and not word.isdigit() and word not in _STOPWORDS
    )


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Intersection over union. 1.0 for identical sets, 0.0 for disjoint ones."""
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


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
    """The full set, all four of ``TECHNICAL_DESIGN.md`` §3.

    Windows are deliberately generous. A detector that fires early costs a real
    result; one that fires late costs some quota. On a 50-evaluation budget,
    eight flat evaluations is a genuine stall rather than bad luck.

    **Nothing in the ablation passes these.** ``run_loop`` takes
    ``detectors=None`` by default and every matrix from Phase 2 onward has run
    that way, for a reason that outranks the design's three-exits structure: the
    evaluation budget has to buy the same number of simulator calls for every
    strategy, and a detector firing at evaluation 12 gives one arm twelve
    against everyone else's twenty. That is the same fairness rule that stops the
    replanner's TERMINATE being obeyed — see ``agent/loop.py``.

    So these exist, are tested, and are for a deliberate single run rather than
    for the comparison. The two that need a critic arrived in Phase 4 and are
    only meaningful for a strategy that has one; on a baseline every diagnosis
    is empty and both return "moving" without ever firing.
    """
    return [
        NoImprovement(window=8),
        DiversityCollapse(window=6, epsilon=0.02),
        ConfidenceDecline(window=6, min_drop=0.15),
        RepeatedDiagnosis(window=4, threshold=0.75),
    ]


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
