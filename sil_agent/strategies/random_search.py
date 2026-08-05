"""Uniform random sampling — the floor that every other strategy must beat.

Named ``random_search.py`` rather than the ``random.py`` of TECHNICAL_DESIGN §12
on purpose: a module named ``random.py`` inside a package shadows the standard
library's ``random`` for anything in that package, and the resulting failure
("module 'random' has no attribute 'Random'") is confusing out of proportion to
the cause.

No intelligence here at all. It exists in Phase 1 to drive the loop so that
persistence can be tested end to end, and survives into Phase 2 as a baseline.
It is a genuinely hard baseline to beat on a small evaluation budget, which is
exactly why it belongs in the comparison.
"""

from __future__ import annotations

import math
import random

from sil_agent.agent.state import Candidate, CandidateSource, Episode, Goal, ParamKind


class RandomSearch:
    """Draws each parameter independently and uniformly from its declared range."""

    @property
    def name(self) -> str:
        return "random_search"

    def propose(
        self,
        goal: Goal,
        history: list[Episode],
        rng: random.Random,
    ) -> Candidate:
        params: dict[str, float | int | str] = {}

        for spec in goal.parameter_space.params:
            match spec.kind:
                case ParamKind.FLOAT:
                    assert spec.bounds is not None
                    low, high = spec.bounds
                    params[spec.name] = rng.uniform(low, high)
                case ParamKind.INT:
                    assert spec.bounds is not None
                    low, high = spec.bounds
                    params[spec.name] = rng.randint(math.ceil(low), math.floor(high))
                case ParamKind.CATEGORICAL:
                    assert spec.choices is not None
                    params[spec.name] = rng.choice(spec.choices)

        return Candidate(
            params=params,
            rationale="uniform random sample from the declared parameter space",
            source=CandidateSource.BASELINE,
        )
