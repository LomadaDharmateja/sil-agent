# Phase 3.5 brief — the memorisation fix and the local provider

Written **before** building. The log (`phase-035.md`) gets written after.

**This phase exists because Phase 4 cannot be run.** Phase 3 ended with two
independent blockers, and neither is a Phase 4 problem to solve in passing:

1. **The model has memorised the benchmarks.** `SingleShotLLM` proposed all
   three of Branin's global minima to four decimal places *before seeing a
   single result*. "Does reflection pay?" is unanswerable on a function where
   the planner already knows the answer — there is nothing to reflect about.
2. **The free tiers cannot carry the experiment.** Measured, not quoted:
   Gemini allows 20 requests per day per model; Cerebras returns 402 on every
   model. Phase 4 needs roughly 2,700 calls. At 20/day that is 135 days.

Phase 4 is the experiment this project exists to run. Spending a short phase
making it *runnable and meaningful* is cheaper than discovering mid-Phase-4 that
the headline number measures recall on a benchmark the agent could not afford to
run five times.

---

## Goal of this phase

A clean baseline: the Phase 3 comparison re-run on functions the model cannot
have seen, using a model that will still exist in two years.

Success is structural, not a win. The agent may do *worse* on a shifted instance
than it did on Branin — in fact it almost certainly will, and that is the point.
A drop is the measurement working.

---

## Part 1 — Prompt leakage

`goal_text` names the function outright:

> "Minimise the Branin function over x1 in [-5, 10] and x2 in [0, 15]."

But scrubbing that string is not the job. The benchmark's identity reaches the
model through **three** channels, and they need different fixes.

| Layer | Where it leaks | Example | Fixable by renaming? |
|---|---|---|---|
| 1. Goal text | `Benchmark.goal_text` → `describe_objective` | "the Branin function" | Yes |
| 2. Metric name | `objective_metric` → rendered as ``Minimise `branin`.`` | `branin`, `hartmann6` | Yes |
| 3. Domain fingerprint | `ParameterSpace` → `describe_space` | `x1 float in [-5, 10]`, `x2 float in [0, 15]` | **No** |

Layer 3 is the one that matters and the one a naive fix misses. A 2-D problem
over exactly `[-5, 10] × [0, 15]` *is* Branin — that asymmetric pair of intervals
identifies it without the name ever being written. Likewise "6-D over the unit
hypercube" is Hartmann-6 to anything that has read the literature. And the domain
cannot be renamed away, because changing it changes the function.

**So items 1 and 2 of the request are one job, not two.** Anonymisation alone
leaves the fingerprint intact; only re-expressing the problem on a neutral,
normalised box removes it — which is exactly what the shifted instances do. The
brief treats them as a single deliverable.

### The scheme

- **Metric** → `objective` for every benchmark.
- **Parameters** → `p1 … pd`.
- **Bounds** → the unit box `[0, 1]^d` for every instance (see Part 2; the
  transform is defined on normalised coordinates anyway, so this is free).
- **Goal text** → states the task and nothing about the landscape:
  *"Minimise `objective` over the parameters below. The function is unknown; no
  analytic form or optimum is available."*

The originals (`branin`, `hartmann6`, `rosenbrock`) get layers 1 and 2 scrubbed
too — defence in depth, and it keeps one code path. They keep their true bounds,
so they stay recognisable, and the log must say so rather than implying the
originals are now clean.

### Enforcement is a test, not a convention

A convention decays the first time someone adds a benchmark. The check is a test
that renders the **actual planner prompt** for every registered simulator and
asserts no forbidden substring survives — `branin`, `hartmann`, `rosenbrock`,
`ackley`, `rastrigin`, `sphere`, `griewank`, `schwefel`, `levy`. Rendering the
real prompt rather than inspecting the dataclass is deliberate: the leak is
whatever reaches the model, not whatever a field is called.

### One consequence worth stating

Changing prompt text changes `call_key` (a SHA-256 over role + system + user +
template version), so **Phase 3's cached model calls become unreachable.** No
result is lost and nothing re-runs — the Phase 3 runs are terminated, so the
harness skips them — but the `phase3-llm` numbers can no longer be replayed
against the new prompt. They stand as recorded history. Anything to be replayed
under the new prompts must be re-run under a new experiment name.

---

## Part 2 — Shifted and rotated instances

### The construction

Work in normalised search coordinates `z ∈ [0, 1]^d`, and let `f̂(w) = f(l + w ⊙ (u − l))`
be the original benchmark on normalised coordinates. The instance is

```
g(z) = f̂( R (z − s) )
```

with `R` a uniformly random rotation and `s` a shift, both from a seeded RNG.
This is the BBOB/COCO form the request specifies.

The request notes the optimum is then at `s + R⁻¹x*`. That is correct, and it is
also the wrong way round to *use*, because an arbitrary `s` puts the optimum
wherever it happens to land — including outside the search box, which would make
regret unreportable. So invert it: **choose where the optimum goes, then solve
for the shift.**

```
w*    = normalised coordinates of a published optimiser
z_opt = seeded draw from [0.15, 0.85]^d       # interior, off the wall
s     = z_opt − Rᵀ w*                          # Rᵀ = R⁻¹ for a rotation
```

Then `g(z_opt) = f̂(w*) = f(x*) = known_optimum`, exactly, and the optimum sits at
`z_opt` by construction. Same map, closed form, containment guaranteed.

`R` comes from the QR decomposition of a Gaussian matrix with the sign
correction that makes the distribution Haar-uniform — a genuinely random
rotation, not a lopsided one. Seeded by `Random(f"{benchmark}:{instance_seed}")`,
a separate stream from the run seed.

### Verified, not assumed

The transform evaluates `f` **outside its published domain**, so `known_optimum`
is only still the optimum if `f` does not dip lower out there. That is an
assumption every regret number would silently rest on, so I checked it before
writing this brief — 400 multi-start L-BFGS-B runs over the search box per
instance:

| Benchmark | Published optimum | `g(z_opt)` | Best found in box | Reach outside box |
|---|---|---|---|---|
| branin (i1) | 0.397887 | 0.397887 | 0.397887 | [−0.57, 1.48] |
| branin (i2) | 0.397887 | 0.397887 | 0.397887 | [−0.53, 1.52] |
| hartmann6 (i1) | −3.322370 | −3.322368 | −3.322368 | [−0.99, 1.92] |
| hartmann6 (i2) | −3.322370 | −3.322368 | −3.322368 | [−1.06, 1.83] |
| rosenbrock (i1) | 0.000000 | 0.000000 | 0.000000 | [−0.85, 1.06] |
| rosenbrock (i2) | 0.000000 | 0.000000 | 0.000000 | [−0.77, 1.50] |

Nothing in the box beats the published optimum, and in every case the multi-start
argmin recovered `z_opt` to three decimals. The residual ~2e-06 on Hartmann-6 is
the rounded literal in `known_optimum`, not an error — the existing regret clamp
at zero already handles it.

The reach turned out far milder than feared. I expected `±√d` (≈ ±2.4 for d=6);
it is under ±1.1. Each coordinate of `R(z − s)` is a dot product with a unit row
vector, which concentrates rather than accumulating.

**This becomes a test.** Every registered instance must be checked, not just the
ones I happened to try, because a bad instance seed silently corrupts every
regret number computed from it. Needs `scipy` as a **dev-only** dependency —
test-time correctness gate, not runtime.

### The instances are harder, and that is a finding

Median regret of plain random search at 20 evaluations, 60 trials:

| Benchmark | Original | Instance i1 | i2 | i3 |
|---|---|---|---|---|
| branin | 1.361 | 6.121 (4.5×) | 6.382 (4.7×) | 6.006 (4.4×) |
| hartmann6 | 1.979 | 2.438 (1.2×) | 2.264 (1.1×) | 1.951 (1.0×) |
| rosenbrock | 3336 | 8676 (2.6×) | 5093 (1.5×) | 5995 (1.8×) |

Branin moves most, and the likely reason is structural: the original box contains
**three** global minima, and the transform generally relocates only one of them
into the box. Fewer targets, same budget. (Stated as the plausible mechanism —
it is checkable and worth confirming in the log.)

**Consequence: instance results cannot be compared to any Phase 2 or Phase 3
number.** They are different functions. Every baseline re-runs on the instances —
which the request already asks for, and now there is a number saying why.

### Identity: where the instance seed lives

The instance seed must be recorded, and *where* it is recorded decides whether a
later mistake is possible.

The report groups results by the `simulator` string. Two instances of Branin are
different functions, so if the instance seed lived outside that string, two
instances would pool into one row and average as though they were repeated seeds
of one problem — precisely the pseudoreplication error Phase 2 caught in grid
search. **So the instance belongs in the simulator's identity**: `branin_i1`,
registered explicitly in `BENCHMARKS`.

This also means `Cell.run_id` is untouched. Appending a field to the UUIDv5 name
would re-derive every identifier and orphan every Phase 2 and Phase 3 run.

Registered explicitly rather than parsed out of the name on demand, so
`from_name` stays a lookup and the legal simulator set stays enumerable for the
CLI. The seed is additionally exposed as `Benchmark.instance_seed` and printed in
the report header, so it is recorded structurally and not only as a substring.

### Deliberately not doing

**No shifted variant of `branin_constrained`.** Its constraint is `x1 + x2 ≤ 10`
on raw parameters, and its "the constrained optimum is still 0.397887" property
depends on exactly which of the three minima survive that half-plane. Under
rotation the half-plane means something else and the property does not survive.
Constraint handling is already exercised by the original; a shifted constrained
variant needs the constraint transformed and re-verified, which is a phase of its
own.

### The test the request calls the real deliverable

*"The single-shot control no longer lands on the optimum before seeing any
result."* That cannot be a normal unit test — `CLAUDE.md` requires LLM calls to
be mocked, and a mocked model proves nothing about memorisation. So it splits:

- **A deterministic unit test**, always run: for each instance, assert the
  optimum location `z_opt` is not recoverable from anything the model sees —
  it differs from every published optimiser, and no coordinate in the rendered
  prompt lies within tolerance of it. This tests the *property* that makes recall
  impossible.
- **A recorded live comparison**, marked and excluded by default: run
  `SingleShotLLM` on original Branin and on `branin_i1`, and report the regret of
  the best proposal in each first batch. The Phase 3 number to beat is
  **3.578e-07 on the original**. If the instance figure is orders of magnitude
  worse, memorisation is broken. Because every call goes through the replay
  cache, this is re-runnable for free and the evidence survives.

The second one produces the headline. A single-shot run on an instance that
*still* scores well is not necessarily recall — space-filling is a respectable
strategy at 20 evaluations. Only the original-versus-instance **gap** is
evidence, which is why both halves get run.

---

## Part 3 — The Ollama provider

### Prerequisite, verified on this machine

- GPU: **NVIDIA RTX 3050 Ti Laptop, 4 GB** — matches `TECHNICAL_DESIGN.md` §5.
- Ollama: **not installed.** Needs `winget install Ollama.Ollama` (or the
  installer from ollama.com) and `ollama pull qwen3:4b-q4_K_M` — roughly a 2.5 GB
  download. This is a user action, not something to do silently.

If the exact tag `qwen3:4b-q4_K_M` does not exist in the registry, **stop and
ask.** Substituting another quantisation is not a small change: a different
quantisation is a different model, and the entire reason for pinning it is the
reproducibility claim in §5.

### Local restores what Phase 3 gave up

Phase 3 had to weaken Rule 1: hosted providers batch requests and move models
under stable aliases, so a resumed run could not be byte-identical. Pinned local
weights plus Ollama's `seed` option plus a fixed quantisation put that back:

| Property | Phase 3 (hosted) | Phase 3.5 (local) |
|---|---|---|
| Resume continues at the correct episode | yes | yes |
| The resumed run is byte-identical | **no** | **yes** |
| Two runs with the same seed match | **no** | **yes** |
| Reproducible in two years | **no** | **yes** |

This is the strongest argument for the switch, and it is a better one than cost.
Say it that way round in the log.

### Constrained decoding, and the line it does not cross

Use `POST /api/chat` with `stream: false`, `format: <JSON schema>`, and
`options: {temperature, seed, num_ctx, num_predict}`. The `format` parameter
compiles the schema to a grammar, so the model *cannot* emit invalid JSON.

**It does not satisfy Rule 2, and that must be stated plainly.** A grammar
guarantees syntax, never semantics. `{"params": {"invented_knob": 1}}` is
schema-valid JSON and still a hallucinated parameter. The guard stays exactly
where it is.

Related and easy to "fix" by mistake: `PlannerProposal.params` is
`dict[str, float | int | str]` with **open keys on purpose** — so an invented
parameter reaches the guard and gets *recorded* rather than being silently
retried inside the router. A grammar derived from that schema therefore will not
constrain parameter names. That is correct. It needs a comment saying so, or
someone will tighten it and delete the project's best hallucination metric.

Risk: Ollama's schema→GBNF conversion does not support everything Pydantic emits
(`anyOf`, `additionalProperties` with union values are the likely casualties).
Mitigation: derive a **simplified grammar schema** separately from the validation
schema, with the Pydantic model remaining the sole authority on acceptance, and a
test asserting the two agree on a known-good payload.

### The trap that will cost a day if unhandled: `num_ctx`

Ollama's default context is small (4096 in current builds) and it **truncates
silently** — no error, no warning. The planner prompt grows with history and
`SingleShotLLM`'s prompt is large by design. A truncated prompt loses the history
block, so the model proposes blind and the run *looks fine*: plausible numbers,
no failures, meaningless results. That is the worst failure mode in this project.

Two mitigations, both required:

- Set `num_ctx` explicitly, and **assert the prompt fits before sending.**
  Failing loudly beats a silently degraded run.
- Watch VRAM. Weights are ~2.5 GB of 4 GB; the KV cache comes out of the
  remainder, so context length trades directly against fitting on the GPU.
  `ollama ps` reports GPU versus CPU residency — check once at the start, as §5
  says. A silent CPU fallback turns a four-hour overnight run into a multi-day
  one.

### Qwen3 is a hybrid reasoning model

Phase 3 lost time to reasoning tokens counting against the output ceiling on
Gemini, where they could not be disabled. Qwen3 has the same hazard, but Ollama
exposes `think: false`. Set it, and **verify it took effect** rather than
assuming — the reply should carry no `thinking` content and no `<think>` block.
On a 4 GB card reasoning tokens are also latency, and across a few hundred calls
that is the difference between an overnight run and a weekend one.

### The provider protocol changes

`Provider.generate(..., json_mode: bool)` cannot express a schema. It becomes
`schema: Mapping[str, object] | None`, with the router deriving it from the
caller's Pydantic model via `model_json_schema()`. Ollama uses it for real;
Gemini can map it to `responseSchema`; Cerebras ignores it. Touches the base
protocol, both existing adapters, `DefaultRouter._call_once`, and the test fakes.

`json_mode` was already documented as "a hint that providers implement
inconsistently" — this replaces a hint with a contract for the one provider that
can honour it, without pretending the others do.

Keep the Gemini adapter, per §5. Worth being honest in the log about what it is
now for: at 20 requests/day it is not meaningful **capacity**, it keeps the
cross-provider failover *path* exercised. That is a correctness property and
portfolio content; it is not a backup plan.

### Schema-compliance rate — defining it so it means something

The obvious definition is worthless. "Fraction of replies that parsed as JSON"
will read 100% under constrained decoding, by construction.

**Definition: the fraction of real provider calls whose *first* completion
validated against the caller's Pydantic schema with no repair attempt.**

Three details that decide whether the number is honest:

- **First attempt only.** The router already retries once; counting after repair
  measures the repair loop, not the model.
- **Real calls only.** `CachingRouter` returns stored *validated* objects, so
  including replays would drive the rate to 100% on any re-run. Cache hits are
  excluded from both numerator and denominator.
- **Validation, not parsing.** A grammatically perfect reply that omits a
  required field is non-compliant.

Recording: add `repair_attempts` (integer) to `llm_calls` via a migration; the
router already knows the count. The report aggregates per strategy and model. A
4B model may well score poorly against Gemini here — that is a result about small
models, reported as a number, exactly as §5 asks.

---

## The re-run

New experiment **`phase35-main`**, and the protocol is chosen to match
`TECHNICAL_DESIGN.md` §5's stated Phase 4 scope so that **Phase 4 adds a row
rather than starting a new experiment**:

| | |
|---|---|
| Simulators | `branin_i1`, `hartmann6_i1` |
| Seeds | **5** — non-negotiable per §5; Phase 2 measured the noise floor at ~102% of the mean |
| Evaluations | 20 |
| Strategies | all five: three baselines + `single_shot_llm` + `agent_no_reflection` |
| Model | `qwen3:4b-q4_K_M`, local |

Call budget: `agent_no_reflection` is 2 benchmarks × 5 seeds × 20 evals = **200
calls**; `single_shot_llm` is one call per run = **10**. Baselines are free.
Roughly 210 local calls — trivial locally, and 10 days of Gemini's free tier.
That contrast is the phase's argument in one line.

Two benchmarks rather than four, per §5's *"prefer fewer benchmarks at five seeds
over more benchmarks at one"*. Branin (2-D) and Hartmann-6 (6-D) span the
dimensionality range and Hartmann's instance difficulty barely moved (1.0–1.2×),
making it the cleaner cross-check. Rosenbrock is deferred: its dynamic range
(regret in the thousands) is hard to read at 20 evaluations.

---

## What I expect to go wrong

Written in advance so the log can be honest about which of these actually
happened.

1. **`qwen3:4b` is materially worse at instruction-following than Gemini.**
   Expect more rejections and a compliance rate below 100%. This is a
   measurement, not a failure, and the guard/rejection accounting from Phase 3
   already handles it — the fairness fix means a hallucinating planner still gets
   its 20 simulator calls.
2. **`num_ctx` truncation, silently.** The one that produces believable garbage.
3. **Thinking tokens not actually disabled**, or disabled at a quality cost.
4. **Grammar conversion rejects the Pydantic schema.**
5. **Something spills to CPU** and the run takes 5× as long.
6. **The single-shot control still scores well on the instance**, and I misread
   it as reasoning. Only the original-versus-instance gap is evidence.
7. **The agent loses to TPE on the instances.** Entirely possible, and it is a
   legitimate result — this phase removes an unearned advantage, so scores should
   fall.

---

## Deliverables

- [ ] `docs/phases/phase-035-brief.md` (this file)
- [ ] Anonymised `goal_text`, metric names and parameter names on every benchmark
- [ ] A test rendering the real planner prompt per simulator, asserting no
      benchmark name leaks
- [ ] Shift + rotate instance construction on normalised coordinates, seeded
- [ ] Instances registered as first-class simulators (`branin_i1`, `hartmann6_i1`)
      with `instance_seed` recorded
- [ ] Optimum-preservation test over the search box for every registered instance
      (`scipy`, dev-only)
- [ ] The memorisation test: deterministic property test + marked live comparison
- [ ] `OllamaProvider` with `format`-based constrained decoding, `seed`, explicit
      `num_ctx`, `think: false`
- [ ] Prompt-fits-context assertion that fails loudly
- [ ] `Provider.generate` protocol change from `json_mode` to `schema`
- [ ] `repair_attempts` column on `llm_calls` (migration) and compliance rate in
      the report
- [ ] `.env.example` entries for the Ollama configuration
- [ ] `phase35-main` executed: 5 strategies × 2 instances × 5 seeds × 20 evals
- [ ] `docs/phases/phase-035.md`, six sections, "what went wrong" non-empty
- [ ] One commit for the phase, per the standing convention

## Out of scope

Critic and replanner (Phase 4). Shifted constrained variants. COCO-style
aggregation across multiple instances — a second, legitimate noise axis, but
Phase 2 measured the *seed* floor and mixing the two now would confound them.
Groq. Any paid model.
