# Phase 3 brief — the agent loop, planner only

Written **before** building. The log (`phase-03.md`) gets written after.

**This is the phase where the LLM arrives.** Everything before it was built so that
when the model starts behaving unpredictably, nothing underneath it is in doubt.

---

## Goal of this phase

Get an LLM proposing candidates inside the existing loop, end to end, on free tiers,
without the loop noticing that the proposer changed.

The measure of success is narrow and structural: `AgentNoReflection` is one more
implementation of `Strategy`, it runs through the unmodified harness, and it appears
in the Phase 2 report as another row. No critic yet — that is Phase 4, and keeping it
out is what makes Phase 4 a clean experiment rather than a comparison against a moving
target.

Beating TPE is **not** a goal of this phase. A planner with no feedback loop very
possibly loses to TPE, and that is a legitimate result to record.

---

## Two things that break in this phase, and how to handle them

These are the reason this brief is longer than the last one. Both are structural, and
both are cheaper to decide now than to discover in week two.

### 1. Determinism dies. Rule 1 has to be restated, not abandoned.

Phase 1 and 2 could promise something strong: kill a run, resume it, get a
byte-identical sequence. An LLM cannot promise that. Even at `temperature=0`, providers
batch requests, change model weights under a stable alias, and route across hardware
that reorders floating-point reductions. Identical inputs do not reliably give identical
outputs.

**What Rule 1 still guarantees, and what it stops guaranteeing:**

| Still true | No longer true |
|---|---|
| Resume continues at the correct episode | The resumed run is byte-identical to an uninterrupted one |
| No episode is lost or run twice | Two runs with the same seed match |
| History, best and budget are recomputed from the record | The seed alone determines the run |

Say this explicitly in the log rather than letting a reader assume the Phase 1 property
still holds everywhere. The Phase 1 claim was real; extending it silently to a
non-deterministic component would be dishonest.

**Recover most of it by recording what the model said.** Store the raw request and
response on each episode. This buys three things for one JSONB column:

- **Exact replay** — re-run an episode against stored responses instead of the provider,
  making a debugging session free, instant and offline.
- **Evidence** — the prompts and completions behind every published number, which is
  what makes a benchmark involving an LLM auditable at all.
- **Phase 10's training data** — distilling the planner into a local Qwen3-4B needs
  exactly these traces, collected as a side effect rather than as a separate exercise.

Add a `ReplayCache` that a strategy consults before calling the provider. Make it the
default in tests, so the whole suite runs with no network and no key.

### 2. The Phase 2 protocol does not fit inside a free tier.

Free tiers rate-limit at 10–15 RPM. Planner-only is one call per episode. So:

```
200 episodes × 1 call ÷ 15 RPM        ≈ 14 min per run
2 LLM strategies × 4 benchmarks × 5 seeds = 40 runs
40 × 14 min                            ≈ 9.3 hours
```

And Phase 4 triples the call count. That is not a schedule, it is a wall.

**Decision: introduce a second budget tier, and re-run every baseline in it.**

Keep `phase2-main` at 200 evaluations as the classical-search reference. Add
`phase3-main` at **50 evaluations**, and run all five strategies there — the three
baselines included. Baselines cost nothing, so re-running them at 50 takes minutes and
keeps the fairness rule intact: *within an experiment, every strategy gets the same
number of simulator calls.* Comparing an LLM at 50 evaluations against TPE at 200 would
be indefensible, and cutting TPE's budget only for the comparison would be worse.

50 evaluations also matters on its own terms: it is much closer to the regime the
project claims to be about, where each simulation costs minutes and nobody gets 200 of
them.

---

## What to build

### 1. `services/router.py` — the only door to a provider

```python
class ModelRouter(Protocol):
    def complete(self, role: Role, prompt: Prompt,
                 schema: type[BaseModel]) -> tuple[BaseModel, CostRecord]: ...
```

`CLAUDE.md` is absolute here: **no agent code imports a provider SDK.** The router is
what makes "switch providers" a config change, and it is why Phase 10 can drop in a
local model without touching the planner.

Per `TECHNICAL_DESIGN.md` §5, roles map to tiers — goal parsing, planner and critic to
strong models; replanner to mid; summarise to cheap. Only the planner and goal parser
exist this phase; define the rest of the enum now so the mapping is stable.

### 2. Provider adapters, with retry that is not optional

Start with **two**, not five: one workhorse and one fallback, so the fallback path is
exercised from the beginning rather than written blind in Phase 6.

Every adapter needs **jittered exponential backoff on HTTP 429**. `CLAUDE.md` calls it
mandatory, and at 15 RPM an agent loop hits it constantly.

Two details that turn a retry loop from decorative into correct:

- **Honour `Retry-After` when the provider sends it.** Guessing when the server has
  told you is how a client gets rate-limited harder.
- **Jitter must be real.** Without it, concurrent workers retry in lockstep and
  re-collide — the thundering herd. `sleep(random.uniform(0, min(cap, base * 2**n)))`.

Retries are a router concern, never a planner concern. A 429 must not reach agent code
and must not become an episode.

### 3. `agent/planner.py` — the LLM proposes

Takes `(goal, history, best)`, returns a `Candidate`. Everything the model sees comes
from persisted state; nothing is carried over in memory between calls.

The output path is where Rule 2 gets real:

```
model returns text
  → parse as JSON            → fail: retry once with the error, then reject
  → validate against schema  → fail: retry once with the error, then reject
  → validate(candidate, space) — the Phase 1 guard, unchanged
  → simulator
```

**The guard becomes load-bearing for the first time.** It was written in Phase 1 against
a sampler that could not produce a bad candidate. Now the proposer can hallucinate a
parameter, and `GuardRejection` becomes a real path: recorded as a `ToolError` episode,
counted against `max_rejections`, and — as Phase 2 established — *not* charged to the
evaluation budget.

Report the rejection rate in the ablation. A planner whose values are constantly clamped
has not understood the parameter space, and `GuardResult.clamped` already carries that
evidence.

### 4. Goal parsing

The one place the LLM touches the problem specification (`TECHNICAL_DESIGN.md` §2).
Free text in, `Objective` and `Constraints` out — validated against the parameter space
the **simulator** declares. Runs once per run, and `ToySimulator.default_goal()` remains
the deterministic path for baselines.

Invent a parameter here and the run fails at episode 0 rather than episode 50.

### 5. Prompts as versioned files (`sil_agent/prompts/`)

Not f-strings buried in `planner.py`. Templates on disk with a version identifier
recorded in every `CostRecord`, so a result can be traced to the prompt that produced
it. Full A/B testing is Phase 11; the versioning that makes it possible costs nothing
now and is unpleasant to retrofit.

### 6. Two LLM strategies

| Strategy | Calls per run | What it tests |
|---|---|---|
| `SingleShotLLM` | 1 | Ask once for all N candidates, evaluate them, no feedback. **Does looping help at all?** |
| `AgentNoReflection` | N | Planner sees history and best each episode, but no critic. |

`SingleShotLLM` is the control that makes the loop's value measurable, and it is nearly
free. Do not skip it because it looks trivial — without it, "the agent beat random
search" cannot be separated from "the model knows what Branin looks like".

### 7. Stagnation detection (`agent/stagnation.py`)

`TECHNICAL_DESIGN.md` §3, as a pluggable list. Implement the two that need no critic:
no improvement in N episodes (default 8), and candidate diversity collapse in normalised
space. Confidence-decline and repeated-diagnosis detectors need the critic — define the
interface, implement them in Phase 4.

Terminating on stagnation must be honest in the report: a run that stops at episode 30
of 50 has not spent its budget, exactly like grid search exhausting. `EXHAUSTED` and
`STAGNATION` are already distinct reasons.

### 8. Redis run locking

Deferred from Phase 1, which noted that the loop currently detects a collision only
when the episode insert conflicts. Lock on `run_id`, TTL longer than the maximum
episode duration, renewed on heartbeat. An LLM episode can take 30 s, so the TTL that
was theoretical in Phase 1 now needs a real number.

### 9. Cost accounting

Populate `CostRecord` — calls, prompt and completion tokens, euro cost, model name,
prompt version. Per-call token caps to catch runaway generation. The reserve-then-commit
governor stays in Phase 6; this is the measurement it will later enforce against.

### 10. Tests — all mocked, no network, no key

Non-negotiable per `CLAUDE.md`. The suite must stay runnable offline and in CI.

- Backoff: 429 → retry with growing, jittered delays; `Retry-After` honoured; give up
  after the cap; a 429 never surfaces as an episode
- Malformed JSON → one repair attempt → `ToolError`, not a crash
- Schema-valid but hallucinated parameter → `GuardRejection` → rejection counted,
  evaluation budget untouched
- Values outside bounds → clamped, and the clamp recorded
- Planner sees only persisted state — same `(goal, history, best)` produces the same
  prompt
- Stagnation fires on a flat history and does not on an improving one
- Redis lock: second worker on the same `run_id` is refused
- Replay cache returns stored responses and makes no call

---

## Acceptance criteria

1. `AgentNoReflection` completes a 50-evaluation run on a free tier, end to end
2. A 429 storm is survived without a failed run — verified against a mocked provider
   that returns 429 for the first N attempts
3. A hallucinated parameter is rejected, recorded, and does not consume an evaluation
4. `phase3-main` runs all five strategies at 50 evaluations and the report renders
5. Raw prompts and responses are stored, and a run replays from cache with no network
6. Killing a run mid-episode and resuming continues at the correct episode — the
   structural guarantee, explicitly not the byte-identical one
7. The full test suite passes offline with no API key set
8. `ruff` clean, `mypy` clean, `pytest` green

Criterion 7 is the one that keeps this project's CI honest.

---

## Known traps

- **Free-tier structured output is uneven.** Some providers support JSON mode, some
  tool calling, some neither reliably. Do not depend on a provider feature — ask for
  JSON, validate with Pydantic, and treat failure as an ordinary path.
- **Models return prose around JSON.** ```` ```json ```` fences, "Here is my proposal:",
  trailing commentary. Extract, then parse.
- **A model that invents parameters will do it every episode.** With `max_rejections`
  at 50 and a 50-evaluation budget, a badly-prompted planner burns the whole rejection
  allowance and terminates having evaluated nothing. That is correct behaviour and a
  useful signal — make sure the report shows it rather than an empty cell.
- **Token cost grows with history.** Feeding all 50 episodes into every prompt makes
  the last call an order of magnitude more expensive than the first. Decide what the
  planner sees — best-k plus recent-k is the usual answer — and record the choice.
- **Rate limits are per key, not per process.** Two experiments in parallel on one key
  do not go twice as fast; they 429 each other.
- **Never commit a key.** `.env` only, `.env.example` updated. The repository is now
  public, so this stopped being a hypothetical rule.

---

## What this unlocks

Phase 4 — the experiment this project exists to run — needs exactly one thing this
phase does not have: a critic. Everything else (router, retries, prompts, guard under
real load, cost accounting, the 50-evaluation protocol with baselines already run in
it) is in place, so Phase 4 adds the critic and re-runs the ablation against a table
that already exists.

---

## Explicitly out of scope

- Critic and replanner (Phase 4) — including the `Evaluation` prose fields
- Episodic memory across runs (Phase 5)
- Budget governor, OpenTelemetry, Langfuse, MCP, FastAPI (Phase 6)
- Surrogate models (7), multi-objective (8), vehicle simulator (9), distillation (10)
- Prompt A/B testing (Phase 11) — versioning only, no experiment harness

The temptation is the critic, because reflection is the interesting idea. Its value is
only measurable against a planner-only baseline that exists first.

---

## Before starting

This phase needs an API key. `TECHNICAL_DESIGN.md` §5 recommends **Cerebras** as the
development workhorse (~1M tokens/day, no card) with **Gemini Flash-Lite** as the
fallback (1,000 requests/day, reliable structured output). Both are free and neither
requires payment details.

Everything except the live run can be built and tested without one.

---

## Finally

Write `docs/phases/phase-03.md` using the template in `README.md`. All six sections,
including "what went wrong". Commit the phase as a unit, log included.
