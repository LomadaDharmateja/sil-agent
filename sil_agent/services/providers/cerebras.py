"""Cerebras — the development workhorse.

Roughly a million tokens a day on the free tier with no payment details, which
is what makes ``TECHNICAL_DESIGN.md`` §5 nominate it for development. The API is
OpenAI-compatible, so this adapter is a thin translation of one JSON shape into
another.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx

from sil_agent.services.providers.base import Completion, raise_for_status, wrap_transport_errors
from sil_agent.services.retry import PermanentError

CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODELS_URL = "https://api.cerebras.ai/v1/models"

# The largest model this account can reach, which is the right default for the
# STRONG tier the planner and goal parser use.
#
# Confirm against the live catalogue rather than assuming — `GET /v1/models`
# lists what the key can actually use. TECHNICAL_DESIGN §5 names Llama 3.3 70B,
# which this account does not have; a stale model name fails as a 404 that reads
# like a permissions problem.
DEFAULT_MODEL = "gpt-oss-120b"


class CerebrasProvider:
    """OpenAI-compatible chat completions against Cerebras.

    The HTTP client is injected so tests can supply a transport that answers
    without a network — the whole suite has to run offline and with no key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("CEREBRAS_API_KEY", "")
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def name(self) -> str:
        return "cerebras"

    def available_models(self) -> list[str]:
        """What this key can actually use.

        Not called by the loop. It exists because a wrong model name fails as a
        404 whose message ("Model does not exist or you do not have access to
        it") reads like a permissions problem, and the fastest way to tell those
        apart is to ask.
        """
        response = self._client.get(
            CEREBRAS_MODELS_URL, headers={"Authorization": f"Bearer {self._api_key}"}
        )
        raise_for_status(response, self.name)
        return [str(item["id"]) for item in response.json().get("data", [])]

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
            # Raised as permanent so the retry loop does not spend a minute
            # discovering that a missing key is still missing.
            raise PermanentError("cerebras: CEREBRAS_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if schema is not None:
            # `json_object`, not `json_schema`. The OpenAI-compatible strict
            # schema mode is not uniformly supported across Cerebras models, and
            # this is the fallback provider — an unconstrained reply that the
            # router then validates is a better failure mode than a 400 from the
            # thing that is supposed to rescue the run.
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self._client.post(
                CEREBRAS_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            raise wrap_transport_errors(self.name, exc) from exc

        raise_for_status(response, self.name)
        return _parse(response.json(), fallback_model=model)


def _parse(body: dict[str, Any], *, fallback_model: str) -> Completion:
    """Read the OpenAI-shaped response.

    Defensive about `usage`: token counts are for cost accounting, and a
    provider omitting them should degrade the accounting rather than fail the
    call.
    """
    try:
        choices = body["choices"]
        text = choices[0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise PermanentError(f"cerebras: unexpected response shape: {body!r:.300}") from exc

    usage = body.get("usage") or {}
    return Completion(
        text=text,
        model=str(body.get("model") or fallback_model),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
