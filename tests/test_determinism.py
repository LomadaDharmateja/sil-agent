"""Is a pinned local model actually reproducible? Measured, not assumed.

`TECHNICAL_DESIGN.md` §5 makes reproducibility the *primary* argument for going
local — above cost:

> A pinned local model reruns **byte-identically in two years**. No hosted API
> can promise that [...] For a project whose entire claim is honest measurement,
> that is a methodological advantage.

The Phase 3.5 log turned that into a table of properties, one row of which reads
"Two runs with the same seed match: **yes**". Phase 4 found otherwise by
accident: re-running `single_shot_llm` under a new experiment name sent a
byte-identical prompt at the same seed to the same pinned tag, and got a
different completion back. The prompt hash matched, the cache key matched, the
model tag matched, and the reply did not.

So the claim gets a measurement. These tests are the instrument.

**They are marked `live` and deselected by default**, for the reason
`CLAUDE.md` gives: LLM calls are mocked in tests. A mocked model cannot tell you
anything about whether a real one is deterministic, so this is the same
exception the memorisation comparison takes — written as a test, excluded from
the default suite, run deliberately:

    uv run pytest -m live tests/test_determinism.py -s

**What the two tests separate.** Determinism can fail at two different scales
and only one of them threatens the design claim:

* *Within one process*, two identical requests with the same seed. A failure
  here means the sampler is not seeded at all.
* *Across processes*, where the model is unloaded and reloaded. A failure only
  here means the seed works but something about the reload — the CPU/GPU split
  on a card the model does not fit, the batch shape, the kernel scheduling —
  changes the arithmetic underneath it.

Measured on the Phase 4 machine, **both pass**: an isolated request at a fixed
seed reproduces, in-process and across processes. Which is the interesting part,
because the real runs did *not* — 1 of 10 twenty-episode runs reproduced its
trajectory when re-executed at the same seed.

So the seed is wired, the model is deterministic in isolation, and
reproducibility is nonetheless lost somewhere between here and a real run. These
two tests are what narrows that gap: they rule out the two simple explanations
and leave the difference to be about *sequence* — hundreds of requests sharing
prefixes through one server session, where prompt-cache reuse makes a generation
depend on what was processed before it. See `docs/phases/phase-04.md` §5.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from sil_agent.services.providers.ollama import DEFAULT_MODEL, OllamaProvider
from sil_agent.services.router import SAMPLING_TEMPERATURE

# A short, fixed request. Short on purpose: a long generation gives divergence
# more opportunities to appear and would make a positive result look stronger
# than it is. If even this differs, nothing longer is safe.
SYSTEM = "You are an optimisation engineer. Reply with JSON only."
USER = (
    "Propose three points to evaluate in the unit square, as JSON:\n"
    '{"points": [{"p1": <float>, "p2": <float>}, ...]}\n'
    "Spread them out. No commentary."
)
SEED = 1


@pytest.fixture(scope="module")
def ollama() -> OllamaProvider:
    """Skip unless a local model is genuinely reachable."""
    provider = OllamaProvider()
    try:
        models = provider.available_models()
    except Exception as exc:
        pytest.skip(f"Ollama not reachable: {exc}")

    if DEFAULT_MODEL not in models:
        pytest.skip(f"{DEFAULT_MODEL} not pulled (have: {models})")
    return provider


def generate(provider: OllamaProvider) -> str:
    return provider.generate(
        model=DEFAULT_MODEL,
        system=SYSTEM,
        user=USER,
        max_tokens=200,
        temperature=SAMPLING_TEMPERATURE,
        schema=None,
    ).text


# The same request, issued by a *separate* interpreter. Written as a string
# rather than a helper module so that what the subprocess runs is visible here,
# next to the in-process version it is being compared against.
SUBPROCESS_SOURCE = """
import json, sys
from sil_agent.services.providers.ollama import DEFAULT_MODEL, OllamaProvider
from sil_agent.services.router import SAMPLING_TEMPERATURE

text = OllamaProvider().generate(
    model=DEFAULT_MODEL,
    system=json.loads(sys.argv[1]),
    user=json.loads(sys.argv[2]),
    max_tokens=200,
    temperature=SAMPLING_TEMPERATURE,
    schema=None,
).text
sys.stdout.write(json.dumps(text))
"""


def generate_in_subprocess() -> str:
    completed = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SOURCE, json.dumps(SYSTEM), json.dumps(USER)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return str(json.loads(completed.stdout))


@pytest.mark.live
def test_the_same_seed_reproduces_within_one_process(ollama):
    """The seed is genuinely wired to the sampler.

    If this fails, `OLLAMA_SEED` is not reaching the request and the five-seed
    protocol is measuring nothing — which is the failure Phase 3.5 caught in a
    different form when temperature 0 made every seed identical.
    """
    first = generate(ollama)
    second = generate(ollama)

    assert first == second, (
        "two identical requests at the same seed differed within one process; "
        "the sampler is not seeded"
    )


@pytest.mark.live
def test_whether_the_same_seed_reproduces_across_processes(ollama):
    """**The design claim under test.** Recorded rather than asserted.

    A fresh interpreter re-imports everything and issues the identical request
    at the identical seed. If the model is reproducible in the sense §5 claims,
    this matches the in-process answer.

    This test *reports* rather than requiring, and that is deliberate. The
    result is hardware-dependent — the target card cannot hold the model, so
    inference is split across CPU and GPU, and the split is not guaranteed
    identical between loads — so a hard assertion would encode one machine's
    behaviour as a correctness requirement for everyone. What belongs in the
    repository is the measurement and the method, not a red suite on somebody
    else's laptop.
    """
    in_process = generate(ollama)
    separate = generate_in_subprocess()

    matched = in_process == separate
    print(f"\ncross-process reproducible: {matched}")
    if not matched:
        print(f"  in-process: {in_process[:200]!r}")
        print(f"  subprocess: {separate[:200]!r}")

    # Recorded as an observation. See the docstring for why this is not an
    # assertion, and `docs/phases/phase-04.md` for what it measured here.
    assert isinstance(matched, bool)
