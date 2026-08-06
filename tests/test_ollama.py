"""The local provider.

Ollama is the primary from Phase 3.5, and unlike the hosted adapters it carries
real logic: it constrains decoding against a schema, budgets the context window,
and detects two kinds of truncation. All of that is tested here against a mock
transport — no server, no model, no network, per `CLAUDE.md`.

The context tests matter most. Ollama truncates an over-long prompt *silently*,
so a run with an oversized prompt does not fail; it produces plausible numbers
that mean nothing. Everything below exists to make that impossible.
"""

from __future__ import annotations

import json

import httpx
import pytest

from sil_agent.services.providers.ollama import (
    CHARS_PER_TOKEN,
    MIN_GENERATION_TOKENS,
    OllamaProvider,
)
from sil_agent.services.retry import PermanentError, TransientError


def make_provider(
    handler,
    *,
    num_ctx: int = 4096,
    **kwargs,
) -> tuple[OllamaProvider, list[dict]]:
    """A provider wired to a mock transport, plus the list of payloads it sent."""
    sent: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(capture))
    return OllamaProvider(client=client, num_ctx=num_ctx, **kwargs), sent


def reply(
    content: str = '{"params": {"p1": 0.5}}',
    *,
    done_reason: str = "stop",
    prompt_eval_count: int = 400,
    eval_count: int = 50,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "qwen3:4b-q4_K_M",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": done_reason,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
        },
    )


def generate(provider: OllamaProvider, *, system="sys", user="user", max_tokens=1024, schema=None):
    return provider.generate(
        model="qwen3:4b-q4_K_M",
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=0.0,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def test_a_schema_is_sent_as_the_format_parameter():
    """This is what makes the model unable to emit invalid JSON.

    Not a prompt instruction — a grammar. `TECHNICAL_DESIGN` §5 is explicit that
    the schema must be passed, because a 4B model asked politely for JSON is
    materially less reliable than one that physically cannot produce anything
    else.
    """
    provider, sent = make_provider(lambda _: reply())
    schema = {"type": "object", "properties": {"params": {"type": "object"}}}

    generate(provider, schema=schema)

    assert sent[0]["format"] == schema


def test_no_schema_means_no_format_key():
    provider, sent = make_provider(lambda _: reply())
    generate(provider, schema=None)
    assert "format" not in sent[0]


def test_thinking_is_disabled_and_the_seed_is_fixed():
    """The two options that make a local run reproducible and affordable.

    `seed` is what restores the Rule 1 guarantee Phase 3 had to give up on
    hosted providers. `think: false` keeps Qwen3's hybrid reasoning from
    spending the token budget — the failure that cost Phase 3 a day on Gemini,
    where it could not be switched off at all.
    """
    provider, sent = make_provider(lambda _: reply(), seed=42)
    generate(provider)

    assert sent[0]["think"] is False
    assert sent[0]["options"]["seed"] == 42
    assert sent[0]["stream"] is False


def test_the_context_window_is_stated_explicitly():
    """Never left to the default, which is small and silently truncating."""
    provider, sent = make_provider(lambda _: reply(), num_ctx=2048)
    generate(provider)
    assert sent[0]["options"]["num_ctx"] == 2048


# ---------------------------------------------------------------------------
# Budgeting the context
# ---------------------------------------------------------------------------


def test_generation_is_clamped_to_what_the_context_leaves():
    """`max_tokens` is a ceiling, not a reservation.

    `SingleShotLLM` asks for 16,000 tokens — a figure sized for Gemini, where
    hidden reasoning was billed against the same allowance. Here the whole
    context is 4,096. Treating the request as a floor would refuse every
    single-shot call; the fix is to give it whatever actually remains.
    """
    provider, sent = make_provider(lambda _: reply(), num_ctx=4096)

    generate(provider, system="s" * 300, user="u" * 300, max_tokens=16_000)

    prompt_estimate = int(600 / CHARS_PER_TOKEN)
    assert sent[0]["options"]["num_predict"] == 4096 - prompt_estimate


def test_a_modest_request_is_passed_through_untouched():
    provider, sent = make_provider(lambda _: reply(), num_ctx=4096)
    generate(provider, max_tokens=512)
    assert sent[0]["options"]["num_predict"] == 512


def test_a_prompt_that_fills_the_window_is_refused_before_the_call():
    """The most dangerous failure in the project, made loud.

    An over-long prompt is not rejected by Ollama — part of it is dropped and
    the model answers the remainder. The planner's history block is the first
    thing to go, so the model proposes blind while the run looks perfectly
    healthy. Refusing costs one clear error; not refusing costs a whole
    experiment that has to be thrown away after the fact.
    """
    provider, sent = make_provider(lambda _: reply(), num_ctx=1024)
    oversized = "x" * int(1024 * CHARS_PER_TOKEN)

    with pytest.raises(PermanentError, match="truncates the prompt"):
        generate(provider, user=oversized)

    assert sent == [], "the call must not be made at all"


def test_the_refusal_threshold_leaves_room_for_a_real_answer():
    """Just fitting is not enough — there must be room to reply."""
    provider, _ = make_provider(lambda _: reply(), num_ctx=1024)
    # Leaves fewer than MIN_GENERATION_TOKENS to answer in.
    almost_full = "x" * int((1024 - MIN_GENERATION_TOKENS + 10) * CHARS_PER_TOKEN)

    with pytest.raises(PermanentError):
        generate(provider, user=almost_full)


# ---------------------------------------------------------------------------
# Truncation, detected rather than passed on
# ---------------------------------------------------------------------------


def test_a_prompt_that_filled_the_window_is_caught_after_the_fact():
    """The exact check, from the server's own token count.

    The pre-flight estimate is a heuristic on characters. This is authoritative:
    `prompt_eval_count` is what the model actually evaluated, so reaching the
    window means the rest was clipped off before it ever saw it.
    """
    provider, _ = make_provider(lambda _: reply(prompt_eval_count=4096), num_ctx=4096)

    with pytest.raises(PermanentError, match="truncated before the model saw it"):
        generate(provider)


def test_output_truncation_says_so_rather_than_failing_in_the_parser():
    """Phase 3's lesson, applied to a second provider.

    Without this the caller receives half a JSON object and fails with
    "Expecting ',' delimiter at position 964" — an error that describes the
    symptom and hides the cause, sending you to the parser instead of the
    token budget.
    """
    provider, _ = make_provider(lambda _: reply(content='{"params": {"p1"', done_reason="length"))

    with pytest.raises(PermanentError, match="truncated at num_predict"):
        generate(provider)


def test_an_empty_completion_is_an_error_not_an_empty_success():
    provider, _ = make_provider(lambda _: reply(content="   "))

    with pytest.raises(PermanentError, match="empty completion"):
        generate(provider)


# ---------------------------------------------------------------------------
# The response, and failures
# ---------------------------------------------------------------------------


def test_tokens_and_model_are_read_from_the_reply():
    provider, _ = make_provider(
        lambda _: reply(prompt_eval_count=419, eval_count=56), num_ctx=4096
    )

    completion = generate(provider)

    assert completion.text == '{"params": {"p1": 0.5}}'
    assert completion.model == "qwen3:4b-q4_K_M"
    assert completion.prompt_tokens == 419
    assert completion.completion_tokens == 56
    assert completion.total_tokens == 475


def test_a_missing_server_is_transient_so_the_run_waits_rather_than_dying():
    """There is no API key here — the credential is a process on localhost.

    A server still warming up must not kill a run that would have worked five
    seconds later, so a connection failure classifies as transient and the
    existing backoff handles it.
    """

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, _ = make_provider(refuse)

    with pytest.raises(TransientError, match="transport error"):
        generate(provider)


def test_available_models_asks_rather_than_guesses():
    """A wrong tag is a 404 that reads like a permissions problem.

    Phase 3 lost time to exactly that on Cerebras. Asking what is installed is
    faster than guessing what might be.
    """

    def tags(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b-q4_K_M"}]})

    client = httpx.Client(transport=httpx.MockTransport(tags))
    provider = OllamaProvider(client=client)

    assert provider.available_models() == ["qwen3:4b-q4_K_M"]
