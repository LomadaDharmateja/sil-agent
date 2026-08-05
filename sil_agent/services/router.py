"""The model router — the only door to a provider.

``CLAUDE.md`` is absolute: *all access goes through `services/router.py`. Never
call a provider SDK directly from agent code.* Everything above this file asks
for "a completion for the PLANNER role, shaped like this schema" and receives a
validated Pydantic object. It never learns which provider answered, how many
times the request was retried, or whether the reply arrived from a cache.

That boundary buys three things the design depends on:

* Switching or stacking providers is configuration, not code
  (``TECHNICAL_DESIGN.md`` §5).
* Phase 10's local Qwen3-4B is a new adapter, not a change to the planner.
* Retries, rate limits and malformed JSON are handled once, in one place,
  instead of once per call site.

**Rule 2 lives here too.** The model returns text. Text becomes an object only
by surviving `json.loads` and then Pydantic validation against a schema the
caller declared. Anything else is a failure with a name, never a half-parsed
dictionary passed hopefully downstream.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ValidationError

from sil_agent.agent.state import CostRecord, Frozen
from sil_agent.services.providers.base import Completion, Provider
from sil_agent.services.retry import (
    BackoffPolicy,
    PermanentError,
    ProviderError,
    RetriesExhausted,
    RetryRecord,
    call_with_backoff,
)


class Role(StrEnum):
    """What a call is for. Roles map to tiers, tiers map to models.

    The full set is defined now even though Phase 3 only uses two, so the
    mapping is stable when Phase 4 adds the critic and replanner.
    """

    GOAL_PARSER = "GOAL_PARSER"
    PLANNER = "PLANNER"
    CRITIC = "CRITIC"
    REPLANNER = "REPLANNER"
    SUMMARISE = "SUMMARISE"


class Tier(StrEnum):
    STRONG = "STRONG"
    MID = "MID"
    CHEAP = "CHEAP"


# TECHNICAL_DESIGN §5. Goal parsing runs once and poisons everything if wrong;
# the planner and critic are the reasoning-heavy calls; the replanner makes a
# structured choice over the critic's output; summarising is high volume and
# low stakes.
ROLE_TIERS: dict[Role, Tier] = {
    Role.GOAL_PARSER: Tier.STRONG,
    Role.PLANNER: Tier.STRONG,
    Role.CRITIC: Tier.STRONG,
    Role.REPLANNER: Tier.MID,
    Role.SUMMARISE: Tier.CHEAP,
}


class Prompt(Frozen):
    """A rendered prompt, carrying the version of the template behind it.

    ``template_version`` is recorded on every CostRecord so a published number
    can be traced back to the prompt that produced it. Phase 11 A/B tests
    prompts; this is the bookkeeping that makes that possible, and it costs
    nothing now.
    """

    system: str
    user: str
    template_version: str = "unversioned"

    # Optional per-call ceiling on generated tokens, overriding the model's
    # default. Needed because one call in this system is not like the others:
    # the planner emits a single small object, while `SingleShotLLM` emits the
    # entire experiment plan at once and needs roughly fifty times the room.
    # Only tokens actually generated are billed, so a generous ceiling on the
    # rare large call costs nothing.
    max_tokens: int | None = None


class ModelRouter(Protocol):
    """TECHNICAL_DESIGN §5."""

    def complete[T: BaseModel](
        self,
        role: Role,
        prompt: Prompt,
        schema: type[T],
    ) -> tuple[T, CostRecord]: ...


class AllProvidersFailed(ProviderError):
    """Every provider was tried and every one failed.

    Carries all of the failures, not just the last. The first version of this
    code raised whatever error the *last* provider produced, which is the
    fallback — the least interesting one. In practice that meant a run whose
    primary was rate limited reported "cerebras: payment required", sending the
    investigation to the wrong provider entirely while the real cause (Gemini's
    daily quota) stayed invisible. The primary's failure is the one that
    explains the run.
    """

    def __init__(self, failures: Sequence[tuple[str, BaseException]]) -> None:
        self.failures = list(failures)
        detail = "; ".join(f"{name}: {error}" for name, error in failures)
        super().__init__(f"all providers failed -- {detail}")

    @property
    def primary_error(self) -> BaseException:
        return self.failures[0][1]


class LLMOutputError(ProviderError):
    """The model answered, but not with something the schema accepts.

    Deliberately distinct from a transport failure. This is not the provider
    malfunctioning — it is the *model* failing to follow instructions, which is
    an ordinary and expected event that the agent records as a failed episode
    rather than a crash.
    """


@dataclass(frozen=True)
class ModelSpec:
    """Which model to use, and how to call it."""

    provider: str
    model: str
    max_tokens: int = 2048
    # Zero by default. It does not make a model deterministic — providers batch
    # requests and route across hardware that reorders floating-point work — but
    # it does remove the sampling noise that is under our control, which is the
    # most that can honestly be claimed. See docs/phases/phase-03-brief.md.
    temperature: float = 0.0


@dataclass(frozen=True)
class RouterConfig:
    """Tier-to-model mapping, plus the fallback order.

    ``fallback_order`` is the point of having two adapters in Phase 3 rather
    than one. A provider that is rate limited past its retry budget, or is down,
    hands off to the next one instead of failing the run — and because the path
    exists from the beginning, it is exercised rather than written blind.
    """

    tiers: dict[Tier, ModelSpec]
    fallback_order: Sequence[str] = ()
    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)

    # Each provider's own model name, so falling back never sends one vendor's
    # model to another. Without this the fallback reused the primary's model
    # name and asked Cerebras for `gemini-2.5-flash-lite`, which fails as a 404
    # reading "Model does not exist or you do not have access to it" — an error
    # that looks like a permissions problem and is actually a routing bug.
    provider_models: dict[str, str] = field(default_factory=dict)

    # Minimum seconds between calls. Free tiers allow 10-15 requests per minute
    # and the loop has no natural pause, so without pacing every call after the
    # first few arrives to a 429 and the run degenerates into backoff. Waiting
    # 4 seconds on purpose is faster than being told to wait.
    min_interval_s: float = 0.0
    # One repair attempt when the model returns something unparseable. Not more:
    # a model that has misunderstood the schema twice will usually keep doing
    # so, and each attempt costs real quota.
    max_repair_attempts: int = 1


class DefaultRouter:
    """Routes by role, retries transient failures, validates every output."""

    def __init__(
        self,
        providers: dict[str, Provider],
        config: RouterConfig,
        *,
        on_call: Callable[[CallRecord], None] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("router needs at least one provider")
        self._providers = providers
        self._config = config
        self._on_call = on_call
        # Monotonic clock: unaffected by the system clock being adjusted, which
        # would otherwise make pacing either stall or vanish.
        self._last_call_at = 0.0

    def complete[T: BaseModel](
        self,
        role: Role,
        prompt: Prompt,
        schema: type[T],
    ) -> tuple[T, CostRecord]:
        """Get a validated object of type ``schema``, or raise.

        Raises:
            LLMOutputError: the model's reply never satisfied the schema.
            ProviderError: every provider failed.
        """
        spec = self._spec_for(role)
        user_prompt = prompt.user
        last_error: Exception | None = None

        calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        model_used = spec.model

        for attempt in range(self._config.max_repair_attempts + 1):
            completion, used_spec = self._generate(
                spec, Prompt(system=prompt.system, user=user_prompt), role
            )
            calls += 1
            prompt_tokens += completion.prompt_tokens
            completion_tokens += completion.completion_tokens
            model_used = completion.model

            try:
                parsed = parse_into(completion.text, schema)
            except LLMOutputError as exc:
                last_error = exc
                if attempt == self._config.max_repair_attempts:
                    break
                # Show the model its own output and the specific complaint.
                # Repeating the original prompt alone tends to reproduce the
                # original mistake.
                user_prompt = _repair_prompt(prompt.user, completion.text, str(exc))
                continue

            cost = CostRecord(
                calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_eur=0.0,  # free tiers during development
                model=f"{used_spec.provider}:{model_used}@{prompt.template_version}",
            )
            return parsed, cost

        assert last_error is not None
        raise LLMOutputError(
            f"{role.value}: model output failed validation after "
            f"{self._config.max_repair_attempts + 1} attempts: {last_error}"
        )

    def _spec_for(self, role: Role) -> ModelSpec:
        tier = ROLE_TIERS[role]
        spec = self._config.tiers.get(tier)
        if spec is None:
            raise PermanentError(f"no model configured for tier {tier.value}")
        return spec

    def _generate(
        self, spec: ModelSpec, prompt: Prompt, role: Role
    ) -> tuple[Completion, ModelSpec]:
        """Call the preferred provider, falling back through the rest.

        Each provider gets the full backoff budget before the next is tried:
        being rate limited is not a reason to abandon the better model until
        waiting has genuinely failed.
        """
        order = [spec.provider, *[p for p in self._config.fallback_order if p != spec.provider]]
        failures: list[tuple[str, BaseException]] = []

        for provider_name in order:
            provider = self._providers.get(provider_name)
            if provider is None:
                continue

            # A fallback provider does not have the primary's model name, so it
            # uses its own configured one — or is skipped if there is none.
            if provider_name == spec.provider:
                active: ModelSpec | None = spec
            else:
                active = self._fallback_spec(provider_name, spec)
            if active is None:
                continue

            try:
                completion, retries = self._call_once(provider, active, prompt)
            except (RetriesExhausted, PermanentError) as exc:
                failures.append((provider_name, exc))
                continue

            if self._on_call is not None:
                self._on_call(
                    CallRecord(
                        role=role,
                        provider=provider_name,
                        model=completion.model,
                        attempts=retries.attempts,
                        waited_s=retries.total_wait_s,
                        rate_limited=retries.rate_limited,
                        prompt_tokens=completion.prompt_tokens,
                        completion_tokens=completion.completion_tokens,
                    )
                )
            return completion, active

        if not failures:
            raise PermanentError(
                f"no usable provider for {role.value}: none of {order} is registered "
                "with a configured model"
            )
        raise AllProvidersFailed(failures) from failures[0][1]

    def _pace(self) -> None:
        """Wait, if the last call was too recent.

        Deliberate pacing beats reactive backoff on a fixed-rate free tier. At
        15 requests per minute an unpaced loop gets perhaps four calls through
        and then spends its time being refused; waiting four seconds on purpose
        is strictly faster than being told to wait, and it leaves the retry
        budget for genuine failures rather than for self-inflicted ones.
        """
        interval = self._config.min_interval_s
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _call_once(
        self, provider: Provider, spec: ModelSpec, prompt: Prompt
    ) -> tuple[Completion, RetryRecord]:
        """One provider, with the full retry budget.

        A method rather than a closure inside the loop above: a lambda capturing
        `provider` and `spec` from a loop binds them late, so if it were ever
        deferred rather than called immediately it would silently use the last
        provider in the list. Passing them as arguments removes the trap instead
        of relying on it not being sprung.
        """

        def operation() -> Completion:
            self._pace()
            self._last_call_at = time.monotonic()
            return provider.generate(
                model=spec.model,
                system=prompt.system,
                user=prompt.user,
                max_tokens=prompt.max_tokens or spec.max_tokens,
                temperature=spec.temperature,
                json_mode=True,
            )

        return call_with_backoff(operation, policy=self._config.backoff)

    def _fallback_spec(self, provider_name: str, original: ModelSpec) -> ModelSpec | None:
        """The spec to use when falling back, or None if that provider is unusable.

        Returns None rather than guessing a model name. Carrying the primary's
        model across to another vendor is how this asked Cerebras for a Gemini
        model and got a 404 that read like a permissions failure — fifty times
        in one run, each recorded as a rejected episode.
        """
        for spec in self._config.tiers.values():
            if spec.provider == provider_name:
                return spec

        model = self._config.provider_models.get(provider_name)
        if model is None:
            return None

        return ModelSpec(
            provider=provider_name,
            model=model,
            max_tokens=original.max_tokens,
            temperature=original.temperature,
        )


@dataclass(frozen=True)
class CallRecord:
    """Observability for one provider call. Phase 6 turns these into OTel spans."""

    role: Role
    provider: str
    model: str
    attempts: int
    waited_s: float
    rate_limited: bool
    prompt_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# Turning text into an object
# ---------------------------------------------------------------------------

# Models wrap JSON in markdown fences constantly, whatever the instructions say.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be wrapped in prose.

    Three layers, cheapest first. This is not defensive programming for its own
    sake — free-tier models with unreliable JSON modes really do return
    "Here is my proposal:" followed by a fenced block, and treating that as a
    failure would throw away a perfectly good answer.
    """
    stripped = text.strip()
    if not stripped:
        raise LLMOutputError("model returned an empty response")

    # 1. Already clean.
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # 2. Fenced.
    fenced = _FENCE.search(stripped)
    if fenced:
        return fenced.group(1).strip()

    # 3. Prose around an object. Take the outermost braces.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]

    raise LLMOutputError(f"no JSON object found in response: {stripped[:200]!r}")


def parse_into[T: BaseModel](text: str, schema: type[T]) -> T:
    """Parse and validate, or raise LLMOutputError with a message worth showing a model.

    The error text is fed back into the repair attempt, so it has to say what is
    actually wrong — "field 'x1' is required" repairs; "validation failed" does
    not.
    """
    payload = extract_json(text)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(
            f"response was not valid JSON ({exc.msg} at position {exc.pos})"
        ) from exc

    if not isinstance(data, dict):
        raise LLMOutputError(f"expected a JSON object, got {type(data).__name__}")

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(_describe(exc)) from exc


def _describe(error: ValidationError) -> str:
    """Compress a Pydantic error into something a model can act on."""
    parts = []
    for item in error.errors()[:5]:
        location = ".".join(str(piece) for piece in item["loc"]) or "(root)"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


def _repair_prompt(original: str, bad_output: str, complaint: str) -> str:
    return (
        f"{original}\n\n"
        "---\n"
        "Your previous response could not be used.\n\n"
        f"You returned:\n{bad_output[:1000]}\n\n"
        f"The problem: {complaint}\n\n"
        "Return only a single valid JSON object matching the required schema. "
        "No markdown fences, no commentary."
    )


# ---------------------------------------------------------------------------
# Building a router from the environment
# ---------------------------------------------------------------------------


def build_default_router(*, on_call: Callable[[CallRecord], None] | None = None) -> DefaultRouter:
    """Construct the router described by the environment.

    Keys come from ``.env`` and never from code. A provider whose key is absent
    is simply not registered, so a machine with only one key configured works
    without any switch being flipped.
    """
    from sil_agent.services.providers.cerebras import DEFAULT_MODEL as CEREBRAS_MODEL
    from sil_agent.services.providers.cerebras import CerebrasProvider
    from sil_agent.services.providers.gemini import DEFAULT_MODEL as GEMINI_MODEL
    from sil_agent.services.providers.gemini import GeminiProvider

    providers: dict[str, Provider] = {}
    if os.environ.get("CEREBRAS_API_KEY"):
        providers["cerebras"] = CerebrasProvider()
    if os.environ.get("GEMINI_API_KEY"):
        providers["gemini"] = GeminiProvider()

    if not providers:
        raise PermanentError(
            "No LLM provider configured. Set CEREBRAS_API_KEY or GEMINI_API_KEY in .env "
            "(see .env.example)."
        )

    preferred = os.environ.get("SIL_PRIMARY_PROVIDER", "cerebras")
    if preferred not in providers:
        preferred = next(iter(providers))

    # Model per provider, overridable from the environment.
    #
    # This is not a convenience. Gemini's free tier meters
    # `GenerateRequestsPerDayPerProjectPerModel` at **20 requests per day per
    # model** — TECHNICAL_DESIGN §5 assumed 1,000 — so which model a run uses is
    # a quota decision, not a quality one, and it has to be changeable without
    # editing code.
    provider_models = {
        "cerebras": os.environ.get("SIL_CEREBRAS_MODEL") or CEREBRAS_MODEL,
        "gemini": os.environ.get("SIL_GEMINI_MODEL") or GEMINI_MODEL,
    }
    specs = {
        name: ModelSpec(provider=name, model=model) for name, model in provider_models.items()
    }
    primary = specs[preferred]

    # Requests per minute on each free tier, converted to a minimum spacing.
    # Gemini Flash-Lite allows 15/min; Cerebras is more generous but the same
    # pacing costs nothing when the quota is not the constraint.
    rpm = {"gemini": 15.0, "cerebras": 30.0}.get(preferred, 15.0)
    min_interval_s = float(os.environ.get("SIL_MIN_CALL_INTERVAL_S", 60.0 / rpm))

    return DefaultRouter(
        providers=providers,
        config=RouterConfig(
            # Phase 3 has no cheap-tier work, so every tier points at the same
            # model. Phase 4 splits them when the replanner arrives.
            tiers={Tier.STRONG: primary, Tier.MID: primary, Tier.CHEAP: primary},
            fallback_order=[name for name in providers if name != preferred],
            provider_models=provider_models,
            min_interval_s=min_interval_s,
        ),
        on_call=on_call,
    )
