"""Google Gemini — the fallback, and the more reliable of the two at JSON.

Flash-Lite allows 1,000 requests a day at 15 RPM on the free tier. The API shape
is Google's own rather than OpenAI-compatible, which is precisely why it is
worth having as the second adapter: it proves the ``Provider`` protocol actually
abstracts something. A second OpenAI-compatible vendor would have proved nothing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx

from sil_agent.services.providers.base import Completion, raise_for_status, wrap_transport_errors
from sil_agent.services.retry import PermanentError

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# The "-latest" alias rather than a pinned version. Free-tier quota is per
# model, and the pinned `gemini-2.5-flash-lite` exhausted its daily allowance
# during development while the alias still had quota — a 429 that reads as "you
# are going too fast" and actually means "this specific model is done for the
# day". Pinning is the right instinct for reproducibility and the wrong one for
# a free tier; the model that answers is recorded on every CostRecord, so what
# actually served each call is still in the record.
DEFAULT_MODEL = "gemini-flash-latest"


class GeminiProvider:
    """Google's generateContent endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def name(self) -> str:
        return "gemini"

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
        if not self._api_key:
            raise PermanentError("gemini: GEMINI_API_KEY is not set")

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if schema is not None:
            # Only the MIME type, deliberately — not `responseSchema`.
            #
            # Gemini's schema dialect is a narrow subset of OpenAPI, not JSON
            # Schema, and it rejects constructs Pydantic emits routinely
            # (`additionalProperties` on an open dict, some `anyOf` shapes).
            # Translating between the two is a project of its own, and getting
            # it wrong turns the *fallback* provider into a source of 400s —
            # the failover path failing is worse than the failover path being
            # unconstrained. Ollama, the primary from Phase 3.5, uses the
            # schema for real.
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            # Gemini keeps the system prompt in its own field rather than as a
            # message with a role.
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

        try:
            response = self._client.post(
                f"{GEMINI_BASE}/{model}:generateContent",
                json=payload,
                # In a header rather than the query string: a key in a URL ends
                # up in proxy logs and shell history.
                headers={"x-goog-api-key": self._api_key},
            )
        except httpx.HTTPError as exc:
            raise wrap_transport_errors(self.name, exc) from exc

        raise_for_status(response, self.name)
        return _parse(response.json(), fallback_model=model)


def _parse(body: dict[str, Any], *, fallback_model: str) -> Completion:
    """Read Google's response shape.

    Two cases that are not errors but produce no text, and must be reported as
    such rather than as an empty success:

    * ``finishReason: SAFETY`` — the response was blocked.
    * ``finishReason: MAX_TOKENS`` with no parts — the model spent its whole
      allowance on reasoning and emitted nothing.

    Both surface as a permanent error, because retrying an identical request
    would produce an identical non-answer.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise PermanentError(f"gemini: no completion returned ({reason})")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    finish_reason = candidate.get("finishReason")

    if not text:
        raise PermanentError(f"gemini: empty completion (finishReason={finish_reason!r})")

    # Truncation, reported as truncation. Without this check the caller receives
    # half a JSON object and fails with "Expecting ',' delimiter at position
    # 964" — an error that describes the symptom and hides the cause, and that
    # sends you looking at the parser instead of at `max_tokens`. It is a
    # permanent error because retrying an identical request truncates in
    # exactly the same place.
    if finish_reason == "MAX_TOKENS":
        raise PermanentError(
            f"gemini: output truncated at max_tokens after {len(text)} characters. "
            "The reply is incomplete, not malformed — raise max_tokens for this call."
        )

    usage = body.get("usageMetadata") or {}
    return Completion(
        text=text,
        model=str(body.get("modelVersion") or fallback_model),
        prompt_tokens=int(usage.get("promptTokenCount") or 0),
        completion_tokens=int(usage.get("candidatesTokenCount") or 0),
    )
