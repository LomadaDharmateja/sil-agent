"""Recording and replaying model calls.

A ``CachingRouter`` wraps any other router and consults the ``llm_calls`` table
before asking a provider. Identical request, same run, same answer — from the
database, for free, offline.

Three things depend on this, which is why it is not an optimisation:

**Statelessness for batch strategies.** ``SingleShotLLM`` must produce its whole
plan in one call but be re-derivable on every episode without holding it in
memory. Rule 1 forbids the obvious fix of caching the batch on the strategy
object. Asking again and hitting this cache satisfies both constraints: one real
call, and the answer is recovered from persisted state rather than remembered.

**Honest debugging.** Investigating a bad agent run otherwise means paying for it
again and getting *different* answers, because the model is not deterministic. A
replayed run reproduces exactly what happened.

**Evidence.** The prompts and completions behind every published number. A
benchmark involving an LLM is only auditable if the model's real inputs and
outputs survive.

The cache key hashes the request content, so it is content-addressed rather than
positional. A planner that produces the same prompt twice — because history has
not changed, or because a run was resumed — gets the same answer without knowing
anything about caching.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel

from sil_agent.agent.state import CostRecord
from sil_agent.persistence.repo import StoredLLMCall
from sil_agent.services.router import ModelRouter, Prompt, Role, parse_into


class CallStore(Protocol):
    """The slice of the repository this needs.

    Declared here rather than importing ``RunRepository`` so the caching router
    has no dependency on SQLAlchemy, and so tests can substitute a dictionary.
    """

    def record_llm_call(
        self,
        run_id: UUID,
        *,
        call_key: str,
        role: str,
        provider: str,
        model: str,
        request: dict[str, object],
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None: ...

    def load_llm_call(self, run_id: UUID, call_key: str) -> StoredLLMCall | None: ...


def call_key(role: Role, prompt: Prompt) -> str:
    """A stable content hash of one request.

    Includes the template version, so bumping a prompt to v2 does not silently
    serve v1's answers — the whole point of versioning prompts is that a
    different prompt is a different question.

    SHA-256 truncated to 32 hex characters: 128 bits, which makes an accidental
    collision within one run impossible in any practical sense, and fits the
    indexed column comfortably.
    """
    payload = json.dumps(
        {
            "role": role.value,
            "system": prompt.system,
            "user": prompt.user,
            "template_version": prompt.template_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class CachingRouter:
    """Wraps a router, recording every call and replaying identical ones.

    Satisfies the ``ModelRouter`` protocol, so nothing above it changes. The
    planner cannot tell whether an answer came from a provider or from Postgres.
    """

    def __init__(
        self,
        inner: ModelRouter,
        store: CallStore,
        run_id: UUID,
        *,
        offline: bool = False,
    ) -> None:
        """``offline=True`` refuses to call a provider at all.

        That is what makes replay a guarantee rather than a hope: a run replayed
        offline either reproduces from the record or fails loudly, and cannot
        quietly make a fresh call that costs money and returns something
        different.
        """
        self._inner = inner
        self._store = store
        self._run_id = run_id
        self._offline = offline
        self.hits = 0
        self.misses = 0

    def complete[T: BaseModel](
        self,
        role: Role,
        prompt: Prompt,
        schema: type[T],
    ) -> tuple[T, CostRecord]:
        key = call_key(role, prompt)

        cached = self._store.load_llm_call(self._run_id, key)
        if cached is not None:
            self.hits += 1
            # Re-validated rather than trusted: the stored text is raw model
            # output, and the schema it must satisfy is the caller's, which may
            # have changed since it was written.
            parsed = parse_into(cached.response, schema)
            return parsed, CostRecord(
                calls=0,  # nothing was spent; a replay is free
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
                cost_eur=0.0,
                model=f"{cached.provider}:{cached.model} (replayed)",
            )

        if self._offline:
            raise LookupError(
                f"offline replay: no recorded {role.value} call for run {self._run_id} "
                f"with key {key}. The prompt differs from the one that was recorded."
            )

        self.misses += 1
        result, cost = self._inner.complete(role, prompt, schema)

        # Store the *validated* object re-serialised, not the raw completion.
        # The raw text is what the model said; this is what the system accepted,
        # and it is what a replay must reproduce. They differ whenever the router
        # had to repair a malformed reply — and replaying the broken version
        # would replay the repair loop too, which is not the run that happened.
        self._store.record_llm_call(
            self._run_id,
            call_key=key,
            role=role.value,
            provider=_provider_of(cost.model),
            model=cost.model or "unknown",
            request={
                "system": prompt.system,
                "user": prompt.user,
                "template_version": prompt.template_version,
            },
            response=result.model_dump_json(),
            prompt_tokens=cost.prompt_tokens,
            completion_tokens=cost.completion_tokens,
        )
        return result, cost


def _provider_of(model: str | None) -> str:
    """`cerebras:llama-3.3-70b@planner.v1` -> `cerebras`."""
    if not model:
        return "unknown"
    return model.split(":", 1)[0][:32]
