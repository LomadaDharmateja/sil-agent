"""Ollama — the local primary from Phase 3.5.

`TECHNICAL_DESIGN.md` §5 was revised after Phase 3 measured what the free tiers
actually give: Cerebras answers 402 on every model, and Gemini allows **20
requests per day per model**. Phase 4 needs roughly 2,700 calls. Hosted free
tiers cannot carry this project.

**But cost is the weaker argument.** A pinned local model reruns byte-identically
in two years. No hosted API can promise that — providers move models under stable
aliases without notice, and when that happens previously published numbers stop
being reproducible. Phase 3 had to weaken Rule 1 for exactly this reason. Running
locally gives it back:

===========================================  ==========  ============
Property                                     Phase 3     Phase 3.5
===========================================  ==========  ============
Resume continues at the correct episode      yes         yes
The resumed run is byte-identical            **no**      **yes**
Two runs with the same seed match            **no**      **yes**
Reproducible in two years                    **no**      **yes**
===========================================  ==========  ============

Three things this adapter does that the hosted ones cannot
----------------------------------------------------------

**Constrained decoding.** ``format`` takes a JSON Schema, which Ollama compiles
to a grammar. The model physically cannot emit non-conforming JSON. This is not
the same as satisfying Rule 2 — see ``providers/base.py`` — but it removes the
single largest failure mode of a 4B model.

**A real seed.** ``options.seed`` makes sampling reproducible.

**An honest context window.** ``num_ctx`` is set explicitly, because Ollama's
default is small and it **truncates silently** rather than erroring. A truncated
planner prompt loses its history block, so the model proposes blind and the run
*looks fine* — plausible numbers, no failures, meaningless results. That is the
worst failure mode available to this project, so it is detected twice: estimated
before the call, and confirmed exactly afterwards from ``prompt_eval_count``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx

from sil_agent.services.providers.base import Completion, raise_for_status, wrap_transport_errors
from sil_agent.services.retry import PermanentError

DEFAULT_HOST = "http://localhost:11434"

# Pinned including the quantisation, per TECHNICAL_DESIGN §5. A bare `qwen3:4b`
# tag is repointed by upstream without notice, which would destroy the
# reproducibility that motivated moving local in the first place. Qwen3-4B is
# also the Phase 10 distillation target, so this is the same model throughout.
DEFAULT_MODEL = "qwen3:4b-q4_K_M"

# Every token of context is VRAM, so this is a real trade-off rather than a
# number to set generously. Measured on the target card (RTX 3050 Ti, 4 GB),
# loading fresh at each size and timing three planner calls:
#
#     num_ctx   footprint   CPU/GPU split   warm call
#      8192       4.3 GB      45% / 55%       2.8 s
#      4096       3.6 GB      36% / 64%       2.2 s
#      2048       3.3 GB      29% / 71%       2.0 s
#
# Note what this says: the model never fits *entirely* on the card, because the
# desktop is already holding some of it. §5 warns about a silent CPU fallback
# turning an overnight run into a multi-day one — that warning is about a full
# fallback. A partial offload costs about 40%, which is affordable, and the
# honest thing is to report the split rather than claim GPU-only inference.
#
# 4096 is the chosen point: a planner prompt carrying twenty episodes of history
# is around 950 tokens and a single-shot plan around 1,400, so this is roughly
# three times what the work needs, at a better residency than 8192.
DEFAULT_NUM_CTX = 4096

# Characters per token, used for the pre-flight estimate. Deliberately
# pessimistic: English prose runs about 4, but these prompts are dense with
# numbers and punctuation, which tokenise worse. Under-estimating here would
# defeat the check.
CHARS_PER_TOKEN = 3.0

# Refuse rather than generate into a sliver. Below this there is not enough room
# for even a single well-formed proposal, so the call would burn time to return
# something truncated.
MIN_GENERATION_TOKENS = 256


class OllamaProvider:
    """Local inference through Ollama's chat endpoint.

    No API key: the "credential" is a process listening on localhost. That also
    means a missing server is a *connection* failure rather than an auth one,
    which the base module already classifies as transient — so a run started
    before the server finished waking up retries instead of dying.
    """

    def __init__(
        self,
        host: str | None = None,
        *,
        client: httpx.Client | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        seed: int = 0,
        think: bool = False,
        # Generous: a local call costs electricity, not quota, and a 4B model on
        # a laptop GPU is slow enough that a timeout tuned for a hosted API
        # would abort perfectly healthy generations.
        timeout_s: float = 600.0,
    ) -> None:
        self._host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)
        self._num_ctx = num_ctx
        self._seed = seed
        self._think = think

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def num_ctx(self) -> int:
        return self._num_ctx

    def available_models(self) -> list[str]:
        """What has actually been pulled. Asking beats guessing at a tag."""
        try:
            response = self._client.get(f"{self._host}/api/tags")
        except httpx.HTTPError as exc:
            raise wrap_transport_errors(self.name, exc) from exc
        raise_for_status(response, self.name)
        return [str(item["name"]) for item in response.json().get("models", [])]

    def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: Mapping[str, Any] | None,
    ) -> Completion:
        num_predict = self._budget_for(system, user, max_tokens)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Qwen3 is a hybrid reasoning model. Phase 3 lost a day to reasoning
            # tokens being billed against the output ceiling on Gemini, where
            # they could not be switched off; here they can. Left on, they are
            # also latency, and across a few hundred calls that is the
            # difference between an overnight run and a weekend one.
            "think": self._think,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": self._num_ctx,
                # The reason a local model restores Rule 1.
                "seed": self._seed,
            },
        }
        if schema is not None:
            payload["format"] = dict(schema)

        try:
            response = self._client.post(f"{self._host}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise wrap_transport_errors(self.name, exc) from exc

        raise_for_status(response, self.name)
        return self._parse(response.json(), fallback_model=model)

    def _budget_for(self, system: str, user: str, max_tokens: int) -> int:
        """How many tokens may be generated, given what the prompt already costs.

        Two different things are happening here, and conflating them was a bug.

        **The prompt must fit, or the call is refused.** Ollama does not reject an
        over-long prompt; it drops part of it and answers anyway. The planner's
        prompt grows with history, so this is a live risk — and a silently
        truncated prompt produces a run that looks entirely normal and means
        nothing.

        **The caller's ``max_tokens`` is a ceiling, not a reservation.**
        ``SingleShotLLM`` asks for 16,000 tokens, a figure sized for Gemini,
        where the model's hidden reasoning was billed against the same
        allowance and could not be switched off. Here reasoning is off and the
        whole context is 4,096, so treating that request as a floor would refuse
        every single-shot call. It is clamped to what actually remains, and a
        reply that still runs out of room is caught by the ``done_reason``
        check rather than passed on half-formed.
        """
        estimated = int((len(system) + len(user)) / CHARS_PER_TOKEN)
        available = self._num_ctx - estimated

        if available < MIN_GENERATION_TOKENS:
            raise PermanentError(
                f"ollama: prompt is about {estimated} tokens, leaving {available} of "
                f"num_ctx={self._num_ctx} to answer in. Ollama truncates the prompt "
                "silently rather than failing, so this is refused here instead. "
                "Raise OLLAMA_NUM_CTX (costs VRAM) or shorten the prompt."
            )

        return min(max_tokens, available)

    def _parse(self, body: dict[str, Any], *, fallback_model: str) -> Completion:
        message = body.get("message") or {}
        text = str(message.get("content") or "")

        prompt_tokens = int(body.get("prompt_eval_count") or 0)
        completion_tokens = int(body.get("eval_count") or 0)

        # The exact check, after the fact. `prompt_eval_count` is the number of
        # tokens actually evaluated, so if it has reached the window the prompt
        # was clipped to fit and whatever fell off never reached the model.
        if prompt_tokens >= self._num_ctx:
            raise PermanentError(
                f"ollama: prompt filled the entire {self._num_ctx}-token context "
                f"({prompt_tokens} evaluated), so it was truncated before the model saw it. "
                "The reply is an answer to a different question than the one asked."
            )

        # Ollama reports "length" when generation hit num_predict. Same reasoning
        # as the Gemini adapter's MAX_TOKENS check: without this the caller gets
        # half a JSON object and a parser error that describes the symptom and
        # hides the cause.
        if body.get("done_reason") == "length":
            raise PermanentError(
                f"ollama: output truncated at num_predict after {len(text)} characters. "
                "The reply is incomplete, not malformed — raise max_tokens for this call."
            )

        if not text.strip():
            raise PermanentError(
                f"ollama: empty completion (done_reason={body.get('done_reason')!r}). "
                "With `think` enabled the model can spend its whole allowance reasoning."
            )

        return Completion(
            text=text,
            model=str(body.get("model") or fallback_model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
