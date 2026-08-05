"""The model router.

No network, no API key. Providers are fakes that return whatever a test needs
them to, including the ugly things real free-tier models actually return:
markdown fences, prose around the JSON, and output that parses but does not
validate.

This is where Rule 2 is enforced — text becomes an object only by surviving
`json.loads` and then Pydantic — so the failure paths matter more than the happy
one.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from sil_agent.services.providers.base import Completion
from sil_agent.services.retry import (
    BackoffPolicy,
    PermanentError,
    RetriesExhausted,
    TransientError,
)
from sil_agent.services.router import (
    AllProvidersFailed,
    DefaultRouter,
    LLMOutputError,
    ModelSpec,
    Prompt,
    Role,
    RouterConfig,
    Tier,
    extract_json,
    parse_into,
)


class Proposal(BaseModel):
    """A stand-in for the planner's output schema."""

    x1: float
    x2: float
    rationale: str = ""


class FakeProvider:
    """Returns scripted replies, and remembers what it was asked."""

    def __init__(self, name: str, replies: list[str | Exception]) -> None:
        self._name = name
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.models: list[str] = []
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def generate(self, *, model, system, user, max_tokens, temperature, json_mode) -> Completion:
        self.calls += 1
        self.prompts.append(user)
        self.models.append(model)
        reply = self._replies.pop(0) if self._replies else '{"x1": 0, "x2": 0}'
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply, model=model, prompt_tokens=10, completion_tokens=5)


def build_router(providers: dict[str, FakeProvider], *, primary: str = "main", **kwargs):
    config = RouterConfig(
        tiers={
            tier: ModelSpec(provider=primary, model="test-model")
            for tier in (Tier.STRONG, Tier.MID, Tier.CHEAP)
        },
        fallback_order=[n for n in providers if n != primary],
        backoff=BackoffPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0),
        # Every provider declares its own model. Fallback needs this: without a
        # model registered for a provider, the router skips it rather than
        # sending the primary's model name to a different vendor.
        provider_models={name: f"{name}-model" for name in providers},
        **kwargs,
    )
    return DefaultRouter(providers=dict(providers), config=config)


PROMPT = Prompt(system="you propose numbers", user="propose one", template_version="v1")


# ---------------------------------------------------------------------------
# Extracting JSON from what models actually send
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_x1"),
    [
        ('{"x1": 1.5, "x2": 2.0}', 1.5),
        ('```json\n{"x1": 1.5, "x2": 2.0}\n```', 1.5),
        ('```\n{"x1": 1.5, "x2": 2.0}\n```', 1.5),
        ('Here is my proposal:\n{"x1": 1.5, "x2": 2.0}\nHope that helps!', 1.5),
        ('  \n {"x1": 1.5, "x2": 2.0}  \n ', 1.5),
    ],
)
def test_extracts_json_from_realistic_replies(raw, expected_x1):
    """Free-tier models wrap JSON in prose and fences whatever the prompt says.

    Treating that as a failure would throw away a perfectly good answer.
    """
    assert parse_into(raw, Proposal).x1 == expected_x1


def test_empty_response_is_an_error_not_an_empty_object():
    with pytest.raises(LLMOutputError, match="empty"):
        extract_json("   ")


def test_prose_with_no_json_is_an_error():
    with pytest.raises(LLMOutputError, match="no JSON object"):
        extract_json("I'm sorry, I can't help with that.")


def test_a_bare_json_array_is_rejected_at_extraction():
    """The schema expects an object. A list is not one, and coercing would guess.

    A bare array never reaches the parser: extraction looks for braces and finds
    none, so it fails one step earlier than the fenced case below.
    """
    with pytest.raises(LLMOutputError, match="no JSON object"):
        parse_into("[1, 2, 3]", Proposal)


def test_a_fenced_json_array_is_rejected_at_parsing():
    """The other route to the same refusal.

    A fenced array extracts cleanly and parses as valid JSON, so it gets as far
    as the type check — which is why that check is not dead code.
    """
    with pytest.raises(LLMOutputError, match="expected a JSON object, got list"):
        parse_into('```json\n[1, 2, 3]\n```', Proposal)


def test_validation_error_names_the_offending_field():
    """The message is fed back to the model for repair, so it has to be specific.

    "field required" repairs; "validation failed" does not.
    """
    with pytest.raises(LLMOutputError, match="x2"):
        parse_into('{"x1": 1.0}', Proposal)


# ---------------------------------------------------------------------------
# The repair attempt
# ---------------------------------------------------------------------------


def test_malformed_output_triggers_one_repair_attempt():
    provider = FakeProvider("main", ["not json at all", '{"x1": 1.0, "x2": 2.0}'])
    router = build_router({"main": provider})

    result, cost = router.complete(Role.PLANNER, PROMPT, Proposal)

    assert result.x1 == 1.0
    assert provider.calls == 2
    assert cost.calls == 2, "both calls are charged, not just the successful one"


def test_the_repair_prompt_shows_the_model_its_own_mistake():
    provider = FakeProvider("main", ["garbage", '{"x1": 1.0, "x2": 2.0}'])
    build_router({"main": provider}).complete(Role.PLANNER, PROMPT, Proposal)

    repair = provider.prompts[1]
    assert "garbage" in repair, "the model must see what it actually returned"
    assert "propose one" in repair, "and the original instruction"


def test_repair_gives_up_and_raises_a_named_error():
    """Two failures is enough. A model that has misunderstood twice keeps going,
    and each attempt costs real quota."""
    provider = FakeProvider("main", ["nope", "still nope", '{"x1": 1, "x2": 2}'])
    router = build_router({"main": provider})

    with pytest.raises(LLMOutputError, match="failed validation after 2 attempts"):
        router.complete(Role.PLANNER, PROMPT, Proposal)

    assert provider.calls == 2, "it must not keep trying past the configured limit"


# ---------------------------------------------------------------------------
# Rate limits and fallback
# ---------------------------------------------------------------------------


def test_a_rate_limited_provider_is_retried_not_abandoned():
    """Being told to slow down is not a reason to switch to a worse model."""
    provider = FakeProvider(
        "main",
        [TransientError("429", retry_after=0.0), '{"x1": 1.0, "x2": 2.0}'],
    )
    router = build_router({"main": provider})

    result, _ = router.complete(Role.PLANNER, PROMPT, Proposal)

    assert result.x1 == 1.0
    assert provider.calls == 2


def test_falls_back_to_the_second_provider_when_the_first_is_exhausted():
    primary = FakeProvider("main", [TransientError("429")] * 5)
    backup = FakeProvider("backup", ['{"x1": 9.0, "x2": 9.0}'])
    router = build_router({"main": primary, "backup": backup})

    result, _ = router.complete(Role.PLANNER, PROMPT, Proposal)

    assert result.x1 == 9.0
    assert primary.calls == 3, "the primary got its whole retry budget first"
    assert backup.calls == 1


def test_a_permanent_failure_falls_straight_through_without_retrying():
    primary = FakeProvider("main", [PermanentError("401 bad key")])
    backup = FakeProvider("backup", ['{"x1": 7.0, "x2": 7.0}'])
    router = build_router({"main": primary, "backup": backup})

    result, _ = router.complete(Role.PLANNER, PROMPT, Proposal)

    assert result.x1 == 7.0
    assert primary.calls == 1, "a bad key will still be bad on the sixth attempt"


def test_fallback_never_sends_one_providers_model_to_another():
    """A regression test for a bug that cost fifty episodes in one run.

    Gemini hit its rate limit, fallback engaged, and the fallback carried
    Gemini's model name across to Cerebras — which answered 404 "Model does not
    exist or you do not have access to it". That reads like a permissions
    problem and is actually a routing bug, and every one of those failures was
    recorded as a rejected episode.

    With no model configured for the fallback provider, it must be skipped.
    """
    primary = FakeProvider("main", [TransientError("429")] * 5)
    backup = FakeProvider("backup", ['{"x1": 5.0, "x2": 5.0}'])

    config = RouterConfig(
        tiers={t: ModelSpec(provider="main", model="main-model") for t in Tier},
        fallback_order=["backup"],
        backoff=BackoffPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        provider_models={"main": "main-model", "backup": "backup-model"},
    )
    router = DefaultRouter({"main": primary, "backup": backup}, config)

    router.complete(Role.PLANNER, PROMPT, Proposal)

    assert backup.models == ["backup-model"], "the fallback used its own model"


def test_a_fallback_provider_with_no_configured_model_is_skipped():
    """Better to skip than to guess a model name and get a confusing 404."""
    primary = FakeProvider("main", [TransientError("429")] * 5)
    unknown = FakeProvider("mystery", ['{"x1": 1.0, "x2": 1.0}'])

    config = RouterConfig(
        tiers={t: ModelSpec(provider="main", model="main-model") for t in Tier},
        fallback_order=["mystery"],
        backoff=BackoffPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        provider_models={"main": "main-model"},  # nothing for "mystery"
    )
    router = DefaultRouter({"main": primary, "mystery": unknown}, config)

    with pytest.raises(AllProvidersFailed) as caught:
        router.complete(Role.PLANNER, PROMPT, Proposal)

    assert unknown.calls == 0, "a provider with no known model is never called"
    assert isinstance(caught.value.primary_error, RetriesExhausted)


def test_pacing_waits_between_calls():
    """Deliberate spacing beats reactive backoff on a fixed-rate free tier.

    At 15 requests per minute an unpaced loop gets a few calls through and then
    spends its time being refused.
    """
    waits: list[float] = []
    provider = FakeProvider("main", ['{"x1": 1, "x2": 2}'] * 3)
    config = RouterConfig(
        tiers={t: ModelSpec(provider="main", model="m") for t in Tier},
        backoff=BackoffPolicy(max_attempts=2, base_delay_s=0.0),
        min_interval_s=4.0,
    )
    router = DefaultRouter({"main": provider}, config)

    import time as time_module

    real_sleep = time_module.sleep
    time_module.sleep = waits.append  # type: ignore[assignment]
    try:
        router.complete(Role.PLANNER, PROMPT, Proposal)
        router.complete(Role.PLANNER, PROMPT, Proposal)
    finally:
        time_module.sleep = real_sleep  # type: ignore[assignment]

    assert waits, "the second call should have been paced"
    assert waits[0] <= 4.0


def test_every_provider_failing_reports_all_of_them_primary_first():
    """The failure that explains the run is the primary's, not the fallback's.

    A regression test for an hour of misdiagnosis: the primary was rate limited
    and the fallback answered "payment required", so that was the error raised
    and recorded on every failed episode. It pointed the investigation at the
    wrong provider entirely while the real cause stayed invisible.
    """
    primary = FakeProvider("main", [PermanentError("401 primary is the real problem")])
    backup = FakeProvider("backup", [PermanentError("403 fallback noise")])
    router = build_router({"main": primary, "backup": backup})

    with pytest.raises(AllProvidersFailed) as caught:
        router.complete(Role.PLANNER, PROMPT, Proposal)

    message = str(caught.value)
    assert "401 primary is the real problem" in message
    assert "403 fallback noise" in message, "the fallback's failure is kept too, not dropped"
    assert "401" in str(caught.value.primary_error), "primary_error is the one to act on"


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_cost_record_carries_tokens_provider_and_prompt_version():
    """The prompt version is in the model string so a published number can be
    traced back to the template that produced it."""
    router = build_router({"main": FakeProvider("main", ['{"x1": 1, "x2": 2}'])})

    _, cost = router.complete(Role.PLANNER, PROMPT, Proposal)

    assert cost.calls == 1
    assert cost.prompt_tokens == 10
    assert cost.completion_tokens == 5
    assert cost.model is not None
    assert "cerebras" not in cost.model  # it used the fake, not a real provider
    assert "@v1" in cost.model


def test_tokens_accumulate_across_repair_attempts():
    provider = FakeProvider("main", ["bad", '{"x1": 1, "x2": 2}'])
    _, cost = build_router({"main": provider}).complete(Role.PLANNER, PROMPT, Proposal)

    assert cost.prompt_tokens == 20, "a wasted call still cost tokens"
    assert cost.completion_tokens == 10


def test_roles_map_to_configured_tiers():
    from sil_agent.services.router import ROLE_TIERS

    assert ROLE_TIERS[Role.PLANNER] is Tier.STRONG
    assert ROLE_TIERS[Role.GOAL_PARSER] is Tier.STRONG
    assert ROLE_TIERS[Role.REPLANNER] is Tier.MID
    assert ROLE_TIERS[Role.SUMMARISE] is Tier.CHEAP
