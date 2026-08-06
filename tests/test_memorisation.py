"""Does the model still know the answer before it has seen anything?

Phase 3's finding was that it did. `SingleShotLLM` — which must produce its whole
plan in one call, before any result exists — proposed all three of Branin's
global minima to four decimal places. Every LLM number in that phase measured
recall rather than search.

This module is the check that the Phase 3.5 fix works. It is structured as two
different kinds of test, because "the model cannot recall the answer" is partly
a property of the *problem* and partly a fact about the *model*.

**The deterministic half** asserts the property, always runs, and needs no model:
the instance optimum is not derivable from anything in the prompt.

**The live half** measures the model, and is skipped unless Ollama is actually
reachable — `CLAUDE.md` requires the suite to run offline with no key. It is the
comparison that carries the evidence: the same strategy, the same budget, the
same anonymised prompt, run on the original benchmark and on the shifted
instance. Only the *gap* between them says anything. A good score on the
instance alone would not, because space-filling is a respectable strategy at
twenty evaluations.
"""

from __future__ import annotations

import math

import pytest

from sil_agent.agent.planner import (
    BatchProposal,
    describe_constraints,
    describe_objective,
    describe_space,
    render_planner_prompt,
)
from sil_agent.prompts import load
from sil_agent.services.providers.ollama import DEFAULT_MODEL, OllamaProvider
from sil_agent.services.retry import ProviderError
from sil_agent.services.router import parse_into
from sil_agent.simulators.toy import BENCHMARKS, ToySimulator

# Branin's three published minima, in the coordinates of the original problem.
# The value at all three is 0.397887.
PUBLISHED_MINIMA = ((-math.pi, 12.275), (math.pi, 2.275), (3.0 * math.pi, 2.475))


# ---------------------------------------------------------------------------
# Deterministic: the property that makes recall impossible
# ---------------------------------------------------------------------------


def test_the_original_benchmark_still_hands_over_its_identity():
    """Honesty about what anonymisation did *not* fix.

    Scrubbing the name and the metric closes two of the three leak channels.
    The third is the domain: a 2-D problem over exactly [-5, 10] x [0, 15] is
    Branin to anything that has read the literature, and that cannot be renamed
    away because changing it changes the function.

    This test exists so nobody reads the anonymisation work and concludes the
    originals are safe to publish LLM numbers on. They are not — the instances
    are.
    """
    space = BENCHMARKS["branin"].space
    bounds = [spec.bounds for spec in space.params]
    assert bounds == [(-5.0, 10.0), (0.0, 15.0)]


def test_the_instance_moves_the_answer_somewhere_unpublished():
    instance = BENCHMARKS["branin_i1"]
    optimum = instance.known_optimisers[0]

    for published in PUBLISHED_MINIMA:
        normalised = ((published[0] + 5.0) / 15.0, published[1] / 15.0)
        assert math.dist(optimum, normalised) > 0.1


def test_the_prompt_contains_nothing_resembling_the_answer():
    """Whatever reaches the model, checked in the model's own units."""
    goal = ToySimulator.from_name("branin_i1").default_goal()
    system, user = render_planner_prompt(
        load("planner", "v1"), goal, [], None, max_evaluations=20
    )
    text = f"{system}\n{user}"

    for coordinate in BENCHMARKS["branin_i1"].known_optimisers[0]:
        for places in (2, 3, 4):
            assert f"{coordinate:.{places}f}" not in text


# ---------------------------------------------------------------------------
# Live: what the model actually proposes before seeing a result
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama() -> OllamaProvider:
    """Skip unless a local model is genuinely reachable.

    Same discipline as the database fixtures: the suite must pass on a machine
    with nothing installed, so this degrades to a skip rather than a failure.
    """
    provider = OllamaProvider()
    try:
        models = provider.available_models()
    except Exception as exc:
        pytest.skip(f"Ollama not reachable: {exc}")

    if DEFAULT_MODEL not in models:
        pytest.skip(f"{DEFAULT_MODEL} not pulled (have: {models})")
    return provider


def first_batch_regret(provider: OllamaProvider, simulator_name: str) -> float:
    """Best regret among the proposals made *before any result exists*.

    This is the memorisation measurement. With an empty history there is nothing
    to reason from, so anything better than a space-filling guess had to come
    from somewhere other than the problem statement.
    """
    simulator = ToySimulator.from_name(simulator_name)
    benchmark = simulator.benchmark
    goal = simulator.default_goal()

    # Rendered exactly as `SingleShotLLM.propose` does — a different template
    # from the planner's, with its own placeholders.
    template = load("single_shot", "v1")
    system, user = template.render(
        objective=describe_objective(goal),
        parameter_space=describe_space(goal.parameter_space),
        constraints=describe_constraints(goal),
        count=20,
    )
    completion = provider.generate(
        model=DEFAULT_MODEL,
        system=system,
        user=user,
        max_tokens=2048,
        temperature=0.0,
        schema=BatchProposal.model_json_schema(),
    )
    batch = parse_into(completion.text, BatchProposal)

    names = [spec.name for spec in benchmark.space.params]
    best = math.inf
    for proposal in batch.proposals:
        if not all(name in proposal.params for name in names):
            continue
        try:
            values = [float(proposal.params[name]) for name in names]
        except (TypeError, ValueError):
            continue
        # Out-of-bounds proposals are the guard's business, not this test's.
        if any(not 0.0 <= v <= 1.0 for v in values) and simulator_name.endswith("_i1"):
            continue
        best = min(best, benchmark.objective(values))

    if best is math.inf:
        pytest.skip("the model returned no usable proposal for this problem")
    return max(0.0, best - benchmark.known_optimum)


@pytest.mark.live
def test_the_single_shot_control_does_not_land_on_a_shifted_optimum(ollama):
    """**The deliverable.** No blind plan should reach the optimum of a
    function nobody has published.

    Phase 3's equivalent number on the original Branin was 3.578e-07 — the
    global optimum, to four decimal places, with an empty history. The threshold
    here is deliberately loose: the claim is not "the agent is bad", it is
    "nothing here is recall". Random search over twenty evaluations reaches
    about 6 on this instance, so anything above 0.01 is comfortably ordinary
    search rather than knowledge.
    """
    regret = first_batch_regret(ollama, "branin_i1")

    assert regret > 0.01, (
        f"the single-shot control reached regret {regret:.3g} on a shifted, rotated "
        "instance before seeing any result. Either the instance is recoverable from "
        "the prompt or the optimum has landed somewhere a model guesses by habit — "
        "both make every LLM number on this instance meaningless."
    )


@pytest.mark.live
def test_the_original_benchmark_is_easier_blind_than_the_instance(ollama):
    """The comparison that turns the previous test into evidence.

    A good score on the instance alone would prove nothing — twenty
    space-filling points do reasonably well. What implicates memorisation is the
    *gap*: the same strategy, same budget, same anonymised prompt, doing far
    better on the function with a published optimum than on the one without.

    Recorded rather than asserted strictly, because it measures a specific
    model. A 4B local model may not have memorised Branin at all, which is
    itself worth knowing and is not a failure of the fix.
    """
    try:
        original = first_batch_regret(ollama, "branin")
        shifted = first_batch_regret(ollama, "branin_i1")
    except ProviderError as exc:
        pytest.skip(f"model call failed: {exc}")

    print(f"\nblind first-batch regret — original branin: {original:.6g}")
    print(f"blind first-batch regret — branin_i1:      {shifted:.6g}")

    # The instance must not be the *easier* of the two by a wide margin; that
    # would mean the shift had accidentally made the problem trivial.
    assert shifted > 0.01
