"""The provider interface, and the HTTP error mapping every adapter shares.

A ``Provider`` returns text and a token count. It does not retry, does not parse
JSON, and does not know what a ``Candidate`` is — those belong to the router and
the planner respectively. Keeping an adapter this thin is what makes adding one
a small, obviously-correct file.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from sil_agent.agent.state import Frozen
from sil_agent.services.retry import PermanentError, TransientError, parse_retry_after


class Completion(Frozen):
    """One response from a provider. Raw text — parsing happens above."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    """Turns a prompt into text. One implementation per vendor."""

    @property
    def name(self) -> str:
        """Stable identifier, recorded with every call so results are attributable."""
        ...

    def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> Completion:
        """Produce a completion, or raise TransientError / PermanentError.

        ``json_mode`` is a *hint*. Providers implement it inconsistently and some
        ignore it, so the router validates the result either way and treats a
        non-JSON reply as an ordinary outcome rather than an impossibility.
        """
        ...


def raise_for_status(response: httpx.Response, provider: str) -> None:
    """Translate an HTTP status into the retry module's vocabulary.

    The classification is the whole point. A 429 or a 503 is a moment in time
    and is worth waiting out; a 400 or a 401 will fail identically forever, and
    retrying it burns a minute of the budget it is supposed to protect.
    """
    if response.is_success:
        return

    status = response.status_code
    detail = _short_body(response)

    if status == 429:
        raise TransientError(
            f"{provider}: rate limited (429) {detail}",
            retry_after=parse_retry_after(response.headers.get("Retry-After")),
        )

    if status in (408, 409, 425):
        raise TransientError(f"{provider}: retryable client error ({status}) {detail}")

    if status >= 500:
        # Includes 529, which several providers use for "overloaded".
        raise TransientError(f"{provider}: server error ({status}) {detail}")

    raise PermanentError(f"{provider}: request failed ({status}) {detail}")


def _short_body(response: httpx.Response) -> str:
    """A trimmed body for the error message.

    Truncated because a provider error body can be a page of HTML, and this
    string ends up in a stored ToolError that someone will read in a terminal.
    """
    try:
        body = response.text.strip().replace("\n", " ")
    except Exception:  # a body that cannot even be decoded must not mask the status
        return ""
    return body[:300]


def wrap_transport_errors(provider: str, exc: httpx.HTTPError) -> TransientError:
    """Connection failures are transient by nature — nothing was answered.

    A timeout, a reset connection or a DNS blip says nothing about whether the
    request was valid, so the only sensible response is to try again.
    """
    return TransientError(f"{provider}: transport error ({type(exc).__name__}: {exc})")
