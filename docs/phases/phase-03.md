# Phase 3 — the agent loop, planner only

Built 2026-08-05. The phase where the LLM arrives.

## 1. Why this phase exists

Everything before this was built so that when the model starts behaving
unpredictably, nothing underneath it is in doubt. Phase 1 made runs durable,
Phase 2 made them measurable, and both were finished before there was anything
to be tempted by.

The deliverable is narrow on purpose: `AgentNoReflection` is one more
implementation of the `Strategy` protocol, it runs through the unmodified Phase 2
harness, and it appears in the same report as another row. No critic — that is
Phase 4, and keeping it out is what makes Phase 4 a clean experiment rather than
a comparison against a moving target.

Beating TPE was explicitly **not** a goal. A planner with no feedback loop may
well lose, and that is a legitimate result to record.

## 2. What I built

| File | Responsibility |
|---|---|
| `services/retry.py` | Jittered exponential backoff, `Retry-After`, transient vs permanent. |
| `services/providers/base.py` | The `Provider` protocol and the HTTP-status-to-error mapping. |
| `services/providers/cerebras.py` | OpenAI-compatible adapter. |
| `services/providers/gemini.py` | Google's own API shape — deliberately *not* OpenAI-compatible. |
| `services/router.py` | The only door to a provider. Roles, tiers, fallback, JSON validation, pacing. |
| `services/replay.py` | Records every call; replays identical ones; `offline=True` refuses to call out. |
| `services/locks.py` | Redis run lock with a renewable TTL, deferred from Phase 1. |
| `prompts/` | Versioned templates on disk, version recorded on every `CostRecord`. |
| `agent/planner.py` | Proposes from persisted state only. |
| `agent/goal_parser.py` | Free text to `Objective`/`Constraints`, validated against reported metrics. |
| `agent/stagnation.py` | No-improvement and diversity-collapse detectors. |
| `strategies/llm_agent.py` | `AgentNoReflection` and `SingleShotLLM`. |
| `persistence/migrations/…34c09c5364b0` | The `llm_calls` table. |

Changed: `agent/loop.py` (provider failures become episodes, duplicate
detection, stagnation exit, LLM cost per episode), `agent/guards.py`
(`is_duplicate`, `perturb`), `strategies/registry.py` (lazy router, so baselines
still need no API key).

## 3. How it works

### Rule 1, restated honestly

Phases 1 and 2 promised something strong: kill a run, resume it, get a
byte-identical sequence. **An LLM cannot promise that**, even at temperature 0 —
providers batch requests, route across hardware that reorders floating-point
work, and move models under a stable alias.

So the guarantee splits:

| Still true | No longer true |
|---|---|
| Resume continues at the correct episode | The resumed run is byte-identical |
| No episode is lost or run twice | Two runs with the same seed match |
| History, best and budget are recomputed from the record | The seed alone determines the run |

Saying this plainly matters more than the loss itself. Quietly extending the
Phase 1 claim to a non-deterministic component would have been the dishonest
option.

Most of the exactness is recovered by **recording what the model said**. Every
prompt and reply goes into `llm_calls`, keyed by a hash of (role, system, user,
prompt version). A completed run replays with no provider calls at all.

### The replay cache is load-bearing, not a convenience

`SingleShotLLM` has to produce its whole plan in one call but be re-derivable on
every episode without holding it in memory. Rule 1 forbids caching the batch on
the strategy object. Regenerating the prompt and hitting the cache satisfies
both: one real provider call, and the answer recovered from persisted state.

A test asserts exactly that — three episodes, `inner.calls == 1`, two cache hits.

### The guard finally does something

Written in Phase 1 against a sampler that could not produce a bad candidate, it
is now the thing standing between a hallucinating planner and the oracle. A
proposal naming a parameter the simulator does not declare is rejected, recorded
as a `ToolError` episode, counted against the rejection allowance — and
crucially **not** charged to the evaluation budget, which is the Phase 2
fairness fix paying off.

### Duplicate detection, and why the design has it

`TECHNICAL_DESIGN.md` §3 contains a line I had not implemented:

```
if is_duplicate(candidate, history):
    candidate = perturb(candidate, seed)
```

I found out why it is there. See §5.

## 4. Key decisions and trade-offs

**A second budget tier.** Free tiers rate-limit at 10–15 RPM and planner-only is
one call per episode, so the Phase 2 protocol (200 evaluations × 5 seeds × 4
benchmarks) is roughly nine hours of wall-clock for one strategy. Phase 3
introduces `phase3-main` at **50 evaluations** and re-runs *every baseline* in
it. Comparing an LLM at 50 evaluations against TPE at 200 would be indefensible,
and cutting TPE's budget only for the comparison would be worse.

That re-run immediately earned its cost — see §Numbers.

**Two providers, not one, and deliberately different shapes.** Gemini's API is
not OpenAI-compatible, which is precisely why it is the second adapter: a second
OpenAI-compatible vendor would have proved nothing about whether the `Provider`
protocol abstracts anything.

**Pacing beats backoff.** At 15 requests per minute, an unpaced loop gets a few
calls through and then spends its time being refused. The router waits
`60/rpm` seconds between calls on purpose. Waiting four seconds deliberately is
strictly faster than being told to wait, and it leaves the retry budget for
genuine failures rather than self-inflicted ones.

**The planner sees best-k plus recent-k, not everything.** Feeding all fifty
episodes into every prompt makes the last call an order of magnitude more
expensive than the first and buries the useful evidence. Recent *rejections* are
included too: a planner not told it invented a parameter will invent it again.

**What would have been easier but worse:** keeping the Optuna-style
"strategy holds its state" pattern for `SingleShotLLM`. One line shorter, and it
breaks resume silently — the restarted process asks the model again and gets a
different plan.

**What would be better but too expensive right now:** running the LLM rows at
five seeds like the baselines. One `agent_no_reflection` sweep at 5 seeds × 4
benchmarks × 50 evaluations is exactly 1,000 calls — the entire Gemini free daily
quota, with nothing left for a mistake. The LLM rows use **3 seeds**; the report
shows n per cell and the asymmetry is stated rather than hidden.

## 5. What went wrong

This section is longer than usual, and most of it is one lesson: **the failures
were all in the plumbing around the model, never in the model.**

**The design already knew about the failure I hadn't implemented.** The first
real agent run proposed the identical point twelve times out of twelve. Not a
rendering bug — I pulled the stored prompt out of `llm_calls` and it was
correct, showing ten identical results — the model simply re-proposed the
incumbent each turn, with a fluent justification every time: *"it is prudent to
re-evaluate this point to confirm its robustness."* A whole budget spent
measuring one point, with reasoning that reads perfectly well.

`TECHNICAL_DESIGN.md` §3 has `is_duplicate` / `perturb` in its loop pseudocode
and `CandidateSource.PERTURB` has been sitting unused in the enum since Phase 1.
I had skipped both as obviously-unnecessary bookkeeping. They exist for exactly
this. Implementing them fixed it, and the perturbation is deterministic
(`Random(f"{seed}:{idx}:perturb")`, a separate stream so nudging cannot consume
draws a strategy was going to use).

**The fallback sent one provider's model name to another.** Gemini hit its rate
limit, fallback engaged, and `_fallback_spec` carried Gemini's model name across
to Cerebras — which answered `404 Model does not exist or you do not have access
to it`. That reads like a permissions problem and is a routing bug. Fifty
episodes in one run were recorded as rejections because of it. My own comment on
the function claimed it "rejects clearly rather than silently guessing"; it
guessed. It now returns `None` and the provider is skipped.

**The fallback's error masked the primary's, and cost an hour.** When every
provider failed, the router raised whatever the *last* one produced — the
fallback, the least interesting one. So a run whose primary was out of quota
reported `cerebras: payment required`, sending me to the wrong provider entirely
while the real cause stayed invisible. I only found it by instrumenting the
exact CLI code path and timing a single call: 31 seconds, which is Gemini
exhausting its retry budget, then Cerebras failing instantly and overwriting the
error. Now `AllProvidersFailed` carries every failure with the primary first.

**Two model names, both wrong, both failing in misleading ways.**
`TECHNICAL_DESIGN.md` §5 names Llama 3.3 70B for Cerebras; that account's
catalogue is `gpt-oss-120b`, `zai-glm-4.7`, `gemma-4-31b`, and the wrong name
404s with a message about access rather than existence. Then the account turned
out to have no free quota at all (402 on every model), so Cerebras was unusable
regardless. On Gemini, the pinned `gemini-2.5-flash-lite` exhausted its daily
allowance during development and started returning 429 — which reads as "slow
down" and actually meant "this model is done for the day" — while
`gemini-flash-latest` still answered. Lesson: **query the catalogue, and prefer
an alias over a pin on a free tier.** I added `available_models()` to the
Cerebras adapter so the next person can ask instead of guess.

**A background process cannot see a config change made after it starts.** I
appended `SIL_PRIMARY_PROVIDER=gemini` to `.env` and relaunched — but an earlier
run was already in flight with the old environment, and because `run_id` is
deterministic, its two failed episodes were permanently in that run's history.
Resuming would have kept them. I deleted the affected runs rather than let a
misconfiguration contaminate a result. The determinism that makes restart free
also makes bad episodes durable.

**mypy caught a protocol that nothing satisfied.** `RedisLike.set` declared
`-> bool | None`; the real client returns `bool | str | bytes | None`, because
`SET ... GET` returns the previous value. So `redis.Redis` did not satisfy my own
protocol. Both methods now return `object`, which is exactly as much as the two
call sites actually know — they only test truthiness.

**I corrupted a test file with a shell one-liner.** A PowerShell `-replace` to
wrap three long lines read the file as ANSI and wrote it back as UTF-8,
double-encoding six em dashes and adding a BOM. The file still parsed and the
tests still passed, so nothing failed — I only caught it by checking raw bytes.
Repaired at the byte level after confirming the damage was confined to that one
file. Lesson: use the editing tools for editing, not the shell.

**Pytest overrides module-level warning filters.** Optuna's experimental-API
warning was silenced at import in `optuna_tpe.py` and still appeared under
pytest, which installs its own filters around every test. It had to be repeated
in `pyproject.toml`.

**Reasoning tokens count against the output ceiling, and cannot be turned off.**
`SingleShotLLM` asks for the whole plan in one reply, and the reply kept
arriving truncated. The first symptom was `Expecting ',' delimiter at position
964` — a parser error describing the symptom and hiding the cause. Adding
explicit `finishReason == "MAX_TOKENS"` detection turned it into "output
truncated at max_tokens", which pointed straight at the real problem: on
`gemini-flash-latest` the model's internal reasoning is billed against
`maxOutputTokens`, and `thinkingConfig.thinkingBudget: 0` is rejected with a
400. Worse, reasoning does not scale with the reply — the same 50-proposal
request used 1,700 reasoning tokens with a terse prompt and blew past 11,500
with this strategy's full instructions. The ceiling is now budgeted for the
pessimistic case, which costs nothing because only generated tokens are billed.

**The free tier is fifty times smaller than the design assumed.** This is the
one that changed the phase's scope. `TECHNICAL_DESIGN.md` §5 records Gemini
Flash-Lite as "1,000 req/day". The actual meter is
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` with **value: 20** — twenty
requests per day, per model. A single 50-evaluation `agent_no_reflection` run
needs 50 calls and therefore *cannot fit on one model at all*. The quota is
per-model, so several models can be used in sequence (five proved usable, giving
about a hundred requests), but the planned 5-seed × 4-benchmark comparison needs
roughly 1,000 and is simply not reachable on this tier. Cerebras, the intended
workhorse, turned out to have no free quota whatever (402 on every model).

The scope was cut honestly rather than faked: the LLM rows are one benchmark,
one seed, 15 evaluations, with **all three baselines re-run at the identical
budget** so the row is like-for-like. What that demonstrates is that the agent
runs end to end on a free tier, which was the phase's stated goal. What it does
not do is support a statistical claim, and the report says so.

## 6. What this unlocks

Phase 4 — the experiment this project exists to run — needs exactly one thing
this phase does not have: a critic. Everything else is in place.

- `Evaluation` already carries `improved`, `delta_vs_best` and `feasible`,
  computed from the simulator, ready to be *injected into* the critic.
- The router has a `CRITIC` role and a `REPLANNER` role mapped to tiers.
- The 50-evaluation protocol exists with every baseline already run in it, so
  Phase 4 adds `agent_full` and re-runs one experiment.
- The replay cache means a Phase 4 debugging session costs nothing and
  reproduces exactly.
- Stagnation detection is pluggable and two of the four detectors in the design
  are implemented; the other two need critic output and have their interface
  settled.

## Numbers

| Measurement | Value |
|---|---|
| Tests | 214 (83 from Phase 1, 54 from Phase 2, 77 new), green, ~20 s |
| Tests requiring a network or API key | **0** |
| Provider adapters | 2 |
| New tables | 1 (`llm_calls`) |
| Planner call cost | ~405 prompt tokens, ~50–65 completion tokens |
| Live call latency | ~2–5 s |
| Pacing | 4.0 s between calls (15 RPM) |
| Free-tier daily quota, measured | **20 requests per day per model** |
| Model calls for a 15-episode single-shot run | **1** (14 served from the replay cache) |
| LLM spend | €0.00 |

### The LLM row, at 15 evaluations on Branin

One seed, one benchmark — all that the free tier allows. Every baseline re-run at
the identical budget so the row is like-for-like:

| Strategy | Regret | Evaluations | Model calls |
|---|---|---|---|
| **agent_no_reflection** | **0.00000036** | 15 | 15 |
| **single_shot_llm** | **0.00000036** | 15 | **1** |
| random_search | 3.604 | 15 | 0 |
| optuna_tpe | 4.876 | 15 | 0 |
| grid_search | 9.91 | 9 (exhausted) | 0 |

Both LLM strategies reach the global optimum; the classical baselines are seven
orders of magnitude behind. **This result should not be believed, and the next
section explains why.** It is reported at all because reporting it and then
explaining it is the honest thing to do with a number that flatters the thing
being built.

Two mechanisms are worth reading off the table anyway, because they are real
regardless of the memorisation. `single_shot_llm` used **one** model call to fill
fifteen episodes — the replay cache doing exactly what it was designed for. And
`agent_no_reflection` produced two `PERTURB` episodes, visible in the per-seed
appendix, where it re-proposed a point it had already evaluated and deterministic
code moved it.

### The 50-evaluation baselines

Re-running the Phase 2 baselines at the reduced budget was the single most
useful decision of the phase, because it changed the answer:

| Strategy | Branin | Hartmann-6 | Rosenbrock |
|---|---|---|---|
| grid_search | 1.798 | 2.817 | 202,600 |
| random_search | 2.534 ± 3.0 | 1.137 ± 0.43 | 1425 ± 1300 |
| optuna_tpe | **0.530 ± 0.21** | **1.125 ± 0.27** | **1371 ± 1600** |

At 200 evaluations TPE beat random search on Hartmann-6 by 5.7× (0.152 vs
0.866). **At 50 evaluations that advantage is gone** — 1.125 against 1.137, far
inside the seed noise floor measured in Phase 2. TPE needs evaluations to build
its surrogate before it can exploit.

That is a real finding about the comparison, not about the agent, and it is
exactly what would have been missed by comparing an LLM at 50 evaluations
against baselines at 200.

### The finding that matters most: the model has memorised the benchmarks

`SingleShotLLM` is asked for its entire plan before it sees a single result. On
Branin its first three proposals were:

```
x1 = -3.1416, x2 = 12.275   ->  0.397887
x1 =  3.1416, x2 =  2.275   ->  0.397887
x1 =  9.4248, x2 =  2.475   ->  0.397887
```

Those are exactly Branin's three global minima — (−π, 12.275), (π, 2.275),
(3π, 2.475) — to four decimal places, proposed with no feedback whatsoever. The
remaining twelve proposals were ordinary space-filling points. The model did not
search; it recalled.

This is not a defect in the agent, and it is the single most useful thing the
phase produced. It means **any result an LLM strategy posts on Branin,
Hartmann-6 or Rosenbrock is contaminated**: these are textbook functions with
published optima, and the model has read the textbook. "The agent beat random
search" would measure recall, not reasoning.

Three consequences, all of which change later phases:

1. **`SingleShotLLM` has earned its place beyond doubt.** It was included as the
   control that answers "does looping help at all?" It turns out to answer a
   more important question first — "is the model optimising or remembering?" —
   and it answers it in one call. Phase 2's decision to build every baseline
   before the agent is what made this visible immediately.
2. **Phase 9's `VehicleEnergySimulator` is promoted from domain credibility to
   methodological necessity.** A bespoke simulator with no published optimum is
   the only way to measure whether the agent can actually reason about a design
   space. It should arrive before the headline numbers are published, not after.
3. **The Phase 4 experiment needs rewording.** "Does reflection pay?" cannot be
   answered on a memorised benchmark, because a planner that already knows the
   answer has nothing to reflect about. Phase 4 should run on a function the
   model cannot have seen — a randomly shifted and rotated variant of a standard
   benchmark is the cheap version, and it keeps the known-optimum property that
   makes regret reportable.

## Interview angle

I built an LLM agent, ran it against classical optimisers, and it beat them by
seven orders of magnitude — and the most valuable thing I did in the phase was
work out why that number is worthless. The single-shot control proposed all three
of Branin's global minima to four decimal places *before seeing a single result*.
It had memorised the benchmark. Any comparison on a textbook function measures
recall, not reasoning, and I caught it because Phase 2 forced me to build the
no-loop control before building the agent.

That is the answer to "tell me about a result you didn't trust." The supporting
story is that every failure in this phase was in the plumbing rather than the
model: a fallback that sent one vendor's model name to another, an error path
where the fallback's failure masked the primary's and sent me to the wrong
provider for an hour, and a free tier that turned out to be twenty requests a day
per model rather than the thousand the design assumed. Meanwhile the model's own
misbehaviour — re-proposing the same point over and over with a fluent
justification each time — was already anticipated by two lines of the design
document I had skipped as obvious bookkeeping.
