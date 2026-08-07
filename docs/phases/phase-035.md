# Phase 3.5 — the memorisation fix and the local provider

Built 2026-08-06. The phase that made Phase 4 possible to run at all.

## 1. Why this phase exists

Phase 3 ended with a result that could not be believed and an experiment that
could not be afforded. Neither was a Phase 4 problem to fix in passing.

**The model had memorised the benchmarks.** `SingleShotLLM` — which must produce
its entire plan before it sees a single result — proposed all three of Branin's
global minima to four decimal places. Every LLM number in Phase 3 measured
recall, not search. "Does reflection pay?", the question Phase 4 exists to
answer, is unanswerable on a function where the planner already knows the answer:
there is nothing to reflect about.

**The free tiers could not carry the work.** Measured rather than quoted: Gemini
allows 20 requests per day per model, and Cerebras returns 402 on every model.
Phase 4 needs roughly 2,700 calls. At 20 a day that is 135 days.

So this phase does three things: it stops the prompt naming the benchmark, it
moves the problems somewhere nobody has published, and it moves inference onto a
local model.

## 2. What I built

| File | Responsibility |
|---|---|
| `simulators/instances.py` | Shift + rotate construction, seeded, with the optimum placed rather than discovered. |
| `services/providers/ollama.py` | The local provider: constrained decoding, explicit context window, two truncation checks. |
| `tests/test_instances.py` | The transform's correctness gate, and the prompt-anonymity checks. |
| `tests/test_ollama.py` | The adapter's logic, against a mock transport. |
| `tests/test_memorisation.py` | The deliverable: deterministic property tests plus a marked live comparison. |

Changed: `simulators/toy.py` (anonymised metric and goal text; instance
registry), `agent/planner.py` (`render_planner_prompt` extracted so the leak test
inspects the real prompt), `services/providers/base.py` (`json_mode: bool` became
`schema`), `services/router.py` (threads the schema, records repair attempts,
registers Ollama, seeds the sampler), `agent/state.py` (`CostRecord.repair_attempts`),
`eval/metrics.py` and `eval/report.py` (schema-compliance rate),
`eval/harness.py` and `cli.py` (the run seed reaches the model).

## 3. How it works

### Anonymisation: three leak channels, two of them closable

The benchmark's identity reached the model three ways, and it matters that they
are not the same problem.

| Layer | Where | Example | Closable by renaming? |
|---|---|---|---|
| Goal text | `Benchmark.goal_text` | "the Branin function" | yes |
| Metric name | `objective_metric` | ``Minimise `branin`.`` | yes |
| Domain fingerprint | `ParameterSpace` | `x1 in [-5,10]`, `x2 in [0,15]` | **no** |

The third is the one that matters. A 2-D problem over exactly `[-5, 10] × [0, 15]`
*is* Branin to anything that has read the literature, whatever the parameters are
called — and the domain cannot be changed without changing the function. So
anonymising the originals is necessary and **not sufficient**, and a test records
that limitation rather than letting a reader assume otherwise. Only the shifted
instances, posed on the unit cube with neutral names, are genuinely anonymous.

The check is a test that renders the *real* planner prompt for every registered
simulator and asserts no benchmark name survives. Rendering the real prompt
rather than rebuilding it is deliberate: a test that constructs its own copy
keeps passing after the planner starts sending something else.

### The instance construction, and why it is inverted

Work in normalised coordinates `z` in the unit cube, with
`f̂(w) = f(l + w ⊙ (u − l))` the original benchmark on normalised coordinates:

```
g(z) = f̂( R (z − s) )
```

`R` is a uniformly random rotation, `s` a shift. That is the standard BBOB form,
under which the optimum lands at `s + R⁻¹w*`.

**This code solves for `s` instead**, and that is the whole trick. Drawing `s`
first puts the optimum wherever it happens to fall — possibly outside the search
box, which makes the run unwinnable and regret meaningless. So:

```
z_opt = seeded draw from the interior of the cube
s     = z_opt − Rᵀ w*
```

Then `g(z_opt) = f(x*) = known_optimum` exactly, and the optimum is inside the
box by construction rather than by luck.

Worked example — `branin_i1`. The published optimiser `(−π, 12.275)` normalises
to `(0.1224, 0.8183)`. The seeded draw puts the new optimum at `(0.6353, 0.3780)`,
and evaluating the instance there returns `0.397887` — Branin's published optimum,
unchanged, at a location nobody has written down.

### The rotation

From the QR decomposition of a Gaussian matrix, with each column of `Q`
multiplied by the sign of the matching diagonal entry of `R`. Without that
correction the distribution is biased and "randomly rotated" is quietly untrue.
`PCG64` is named explicitly rather than taking `default_rng`'s choice, which is
documented as changeable — an instance whose geometry depends on the installed
NumPy version is not reproducible in the sense this project claims.

### Local inference restores what Phase 3 gave up

Phase 3 had to weaken Rule 1: hosted providers batch requests and move models
under stable aliases, so a resumed run could not be byte-identical. Pinned
weights, a fixed quantisation and Ollama's `seed` put it back.

| Property | Phase 3 (hosted) | Phase 3.5 (local) |
|---|---|---|
| Resume continues at the correct episode | yes | yes |
| The resumed run is byte-identical | **no** | **yes** |
| Two runs with the same seed match | **no** | **yes** |
| Reproducible in two years | **no** | **yes** |

That is a better argument for going local than cost, and it is the one to lead
with.

### Constrained decoding, and the line it does not cross

`format` takes a JSON Schema, which Ollama compiles to a grammar; the model
physically cannot emit non-conforming JSON. **This is not Rule 2.** A grammar
enforces syntax, never meaning: `{"params": {"invented_knob": 1}}` conforms
perfectly and is still a hallucinated parameter. The guard stays exactly where it
was.

Related, and easy to "fix" by mistake: `PlannerProposal.params` has open keys on
purpose, so an invented parameter reaches the guard and is *recorded* rather than
silently retried inside the router. A grammar derived from that schema therefore
does not constrain parameter names. That is correct, and it now says so in a
comment, or someone will tighten it and delete the project's best hallucination
metric.

## 4. Key decisions and trade-offs

**The instance seed lives in the simulator's name.** `branin_i1`, not a separate
column. The report groups by simulator, and two instances of Branin are different
functions — if the seed lived anywhere else, two of them would pool into one row
and be averaged as though they were repeated seeds of one problem. That is
exactly the pseudoreplication Phase 2 caught in grid search. Putting it in the
identity makes the mistake impossible rather than merely unlikely, and it leaves
`Cell.run_id` untouched so no Phase 2 or Phase 3 run is orphaned.

**No shifted variant of the constrained benchmark.** Its constraint is stated on
raw parameters and its known optimum depends on which minima survive the
half-plane; rotation does not preserve either. `make_instance` refuses rather
than producing an instance whose declared optimum is wrong.

**Compliance measured on `CostRecord`, not a new `llm_calls` column.** The brief
specified a migration. `CostRecord` is already persisted per episode, already
carries `calls` — which distinguishes a real call from a replay — and needed one
field. A migration would have bought per-call granularity the report does not
use.

**The context window is 4096, not 8192.** Measured on the target card rather than
guessed; see Numbers. 4096 is roughly three times what the work needs at a better
GPU residency.

**What would have been easier but worse:** keeping temperature at 0. It makes
every run trivially reproducible — and identical across seeds, which destroys the
variance the five-seed protocol exists to measure. See §5.

**What would be better but too expensive right now:** COCO-style aggregation
across several instances per benchmark. Instance variation is a second legitimate
noise axis, but Phase 2 measured the *seed* floor and mixing the two would
confound them. One instance per benchmark, five seeds, as §5 directs.

## 5. What went wrong

**I built a benchmark that was rigged in the agent's favour, and it looked like
success.** The first smoke run scored regret 3×10⁻⁴ in *five* evaluations on a
function the model had supposedly never seen. My first thought was that the
anonymisation had failed. It had not. The episodes read:

```
(0.5, 0.5)  -> 23.17     "starting from the center"
(0.7, 0.3)  -> 18.32
(0.6, 0.4)  ->  0.398    <- the optimum
```

The seeded optimum had landed at `(0.6001, 0.4017)`. The model walked a coarse
grid of round numbers and one of them *was* the answer.

This is worse than a lucky instance. **Language models propose round numbers
constantly; random search and TPE do not.** So an optimum sitting on the grid is
differentially easy for exactly the strategy under test — a systematic bias in
favour of the thing being measured, which would have shown up as the agent
reasoning brilliantly. Optima are now rejection-sampled clear of the 0.05 grid,
and a regression test sweeps every round number and asserts real regret survives.

I nearly kept this number. It was the second time in two phases that a result
flattering the agent turned out to be an artefact, and the tell was the same both
times: it was too good, too early.

**Five seeds were not five seeds.** The first full LLM matrix returned *identical*
results for all five seeds of `single_shot_llm` — 3.432490, five times. At
temperature 0 decoding is greedy and the seed does nothing. Reporting that as
n=5 is precisely the pseudoreplication Phase 2 caught in grid search, where five
identical values produced the most significant p-value attainable at n=5.

`TECHNICAL_DESIGN.md` §5 calls five seeds non-negotiable, and the reason is
variance — a mean with no spread cannot be compared to anything. So the run's
seed now drives the model's sampler at temperature 0.7. Each seed is a genuinely
independent replicate *and* individually reproducible, which is strictly more
than temperature 0 gave. I stopped the running matrix and deleted it rather than
finish collecting a number I already knew was invalid.

**A latent bug from Phase 3: `Prompt.max_tokens` never reached the provider.**
The router rebuilt the prompt as `Prompt(system=..., user=...)` before calling
`_generate`, dropping the per-call token ceiling, which then fell back to the
2048 default. So Phase 3's carefully reasoned `max(16_000, 8_000 + 300 × count)`
budget — the fix for the truncation that cost a day — was never actually in
force. Found by threading the schema through the same call path. Fixed, and the
fake provider now records the `max_tokens` it was given so a test can see it.

**The model never fits entirely in VRAM, at any context size.** `ollama ps`
reported 45%/55% CPU/GPU at `num_ctx=8192` on a 4 GB card. Reducing to 2048 got
it to 29%/71% and no further — the desktop is already holding part of the card.
§5's warning is about a *silent full* fallback turning an overnight run into a
multi-day one; a partial offload costs about 40%, which is affordable. The honest
thing is to report the split rather than claim GPU-only inference, so the measured
table is in Numbers.

**`SingleShotLLM` asks for more tokens than the whole context.** Its 16,000-token
request was sized for Gemini, where hidden reasoning was billed against the same
allowance and could not be disabled. Locally, reasoning is off and the entire
context is 4,096, so treating that request as a reservation refused every
single-shot call. The adapter now treats `max_tokens` as a ceiling and clamps it
to what the prompt leaves — with the prompt itself still refused outright if it
does not fit, because Ollama truncates prompts silently and a truncated planner
prompt produces a run that looks healthy and means nothing.

**The brief got two things wrong.** It called for `scipy` as a new dev-only
dependency; scipy has been a runtime dependency since Phase 2, which I would have
known by reading `pyproject.toml` before writing rather than after. And it
specified an `llm_calls` migration for the compliance metric, which turned out to
be unnecessary. Both are recorded here rather than edited out of the brief: a
pre-build document that has been quietly corrected afterwards is no longer
evidence of what was known beforehand.

**The convergence charts painted two strategies the same colour.** Phase 3's log
predicted the palette would run out; it ran out one phase earlier than expected,
and `colour_for` cycled with `% len(...)` instead of refusing. Five series, three
hues. Found only by rendering the PNG and looking at it — the code was clean and
the validator passed for the three colours it had been given. Detail in Numbers.

**The first constrained-decoding ablation measured almost nothing.** It ran at
temperature 0.0 while the experiment ran at 0.7, and a bug left its
"with history" prompts empty, so it only exercised short opening prompts. It
reported the right answer for the wrong reasons. Redone by replaying the real
recorded prompts out of `llm_calls`; the conclusion held, but it would not have
been worth stating from the first version.

**Two processes wrote to the same run, and the Phase 1 guard was the only thing
that noticed.** The budget sweep's shell script was killed partway through, so I
reissued it — and the *child* `ablate` process had survived the kill. Two
processes were then walking the same experiment. What stopped it being a silent
corruption was a line written in Phase 1:

```
episode 11 of run ef366677-... already exists - another process is writing to
this run. Stopping. (Run locking arrives in Phase 3.)
```

The natural key on `(run_id, idx)` refused the duplicate write. No run was
corrupted, and the two interrupted runs were left `EXECUTING` — which
`derive_termination` maps to `None`, so the harness resumes them rather than
mistaking them for finished. Both properties were designed in Phase 1 against a
hypothetical; this is the first time either was needed in anger.

The irony is that Phase 3 built `services/locks.py` for exactly this and the
harness does not use it — locking is wired for the CLI's single-run path, not for
`execute_cell`. Recorded as the obvious next use of code that already exists.

The operational lesson is smaller and duller: **killing a task does not always
kill its children.** Check for a surviving process before reissuing a long run.

It happened repeatedly, because the budget-80 sweep runs are longer than the
harness that launches them will tolerate — every invocation was cut short after a
few minutes. The sweep still finished, one re-issue at a time, and that is worth
recording as the clearest vindication of Phase 1 in the project so far: **the
80-evaluation cells were assembled across five separate killed processes**, with
no resume flag, no job table and no manual bookkeeping. `ensure_run` loads what
exists, `rehydrate` recomputes position from the episodes table, and the natural
key makes a re-issued command a no-op over the parts already done. The only thing
I had to get right was not starting a second writer.

**The live tests ran by default and broke the offline rule.** `CLAUDE.md`
requires LLM calls to be mocked, and the memorisation comparison is the one thing
that genuinely cannot be — a mocked model proves nothing about what a real one
memorised. Written as a normal test, it ran whenever Ollama happened to be
installed, taking the suite from 22 seconds to two minutes and making it
non-deterministic. It is now marked `live` and deselected by default.

## 6. What this unlocks

Phase 4 can now be both **run** and **believed**, which it could not be before.

- The benchmarks are functions the model cannot have read about, so a difference
  between `agent_no_reflection` and `agent_full` measures reflection rather than
  recall.
- Inference is local and effectively unlimited, so the 5-seed × 2-benchmark
  protocol in §5 is an overnight run rather than 135 days of free-tier quota.
- The protocol here — 20 evaluations, 5 seeds, `branin_i1` and `hartmann6_i1` —
  is deliberately the one §5 specifies for Phase 4, so Phase 4 adds a row to an
  existing comparison instead of starting a new one.
- Every LLM strategy now varies with its seed, so Phase 4's headline claim can
  have an error bar and be tested against the noise floor.
- `schema_compliance` is in the report, so Phase 10's distillation target has a
  baseline to beat.
- **`sil-agent sweep` exists**, so Phase 4's question can be asked at the right
  width from the start: not "does reflection pay?" but "at which budgets does
  reflection pay?". `agent_full` becomes one more curve on the same axes, and
  the crossover it needs to move is already measured at ~37 evaluations.

## Numbers

| Measurement | Value |
|---|---|
| Tests | 305 (301 offline, 2 skipped in high dimension, 2 live and deselected) |
| Tests requiring a network or API key | **0** |
| Suite runtime | ~22 s offline; ~94 s for the two live tests |
| Model | `qwen3:4b-q4_K_M`, pinned including quantisation |
| Model download | 2.6 GB |
| Warm planner call | ~2.2 s |
| Cold start (first call after load) | 9–12 s |
| Model calls in `phase35-main` | 210 (200 planner + 10 single-shot) |
| Schema compliance | 100%, and 100% with the grammar disabled |
| Runs in the database this phase | 170 (50 main + 120 noise) |
| LLM spend | **€0.00** |
| Equivalent on Gemini's free tier | 210 calls ÷ 20 per day = **11 days** |

### Context window versus GPU residency, measured

RTX 3050 Ti Laptop, 4 GB. Loaded fresh at each size, three planner calls timed:

| `num_ctx` | Footprint | CPU/GPU split | Warm call |
|---|---|---|---|
| 8192 | 4.3 GB | 45% / 55% | 2.8 s |
| 6144 | 4.0 GB | 40% / 60% | 2.5 s |
| 4096 | 3.6 GB | 36% / 64% | 2.2 s |
| 3072 | 3.5 GB | 34% / 66% | 2.1 s |
| 2048 | 3.3 GB | 29% / 71% | 2.0 s |

Cold start (first call after load) is 9–12 s. **4096 chosen**: three times the
headroom the work needs, at better residency than 8192.

### The transform preserves the optimum

Verified before the brief was written, by 400 multi-start L-BFGS-B runs over the
search box per instance — the assumption every regret number rests on.

| Benchmark | Published optimum | `g(z_opt)` | Best found in box | Reach outside box |
|---|---|---|---|---|
| branin_i1 | 0.397887 | 0.397887 | 0.397887 | [−0.57, 1.48] |
| hartmann6_i1 | −3.322370 | −3.322368 | −3.322368 | [−0.99, 1.92] |
| rosenbrock_i1 | 0.000000 | 0.000000 | 0.000000 | [−0.85, 1.06] |

Nothing in the box beats the published optimum, and the multi-start argmin
recovers `z_opt` to three decimals in every case. The residual 2×10⁻⁶ on
Hartmann-6 is the rounded literal in the constant, not an error.

I expected the reach outside the original domain to be about `±√d` (≈ ±2.4 at
d=6) and it is under ±1.1: each coordinate of `R(z − s)` is a dot product with a
unit row vector, which concentrates rather than accumulating.

### The instances are harder than the originals

Median regret of plain random search at 20 evaluations, 60 trials:

| Benchmark | Original | i1 | i2 | i3 |
|---|---|---|---|---|
| branin | 1.361 | 6.121 (4.5×) | 6.382 (4.7×) | 6.006 (4.4×) |
| hartmann6 | 1.979 | 2.438 (1.2×) | 2.264 (1.1×) | 1.951 (1.0×) |
| rosenbrock | 3336 | 8676 (2.6×) | 5093 (1.5×) | 5995 (1.8×) |

Branin moves most, most likely because the original box contains three global
minima and the transform relocates only one of them into the box. **Instance
results are therefore not comparable to any Phase 2 or Phase 3 number**, which is
why every baseline was re-run.

### The comparison, on shifted instances, with a local 4B model

`phase35-main`: 5 strategies × 2 instances × 5 seeds × **20 evaluations**. Final
regret, mean ± standard deviation, lower is better.

**The budget in that sentence is not a detail, it is the claim.** Everything
below is about behaviour under a tight budget; none of it says the agent is a
better optimiser than TPE in general, and TPE at 200 evaluations reaches regret
0.0188 — an order of magnitude below anything measured here. See "Rescoping the
claim" below.

| Strategy | branin_i1 | hartmann6_i1 |
|---|---|---|
| **agent_no_reflection** | **0.381 ± 0.23** | **1.052 ± 0.56** |
| grid_search | 2.196 ± 0 | 1.956 ± 0 |
| optuna_tpe | 2.838 ± 2.5 | 1.540 ± 0.64 |
| single_shot_llm | 4.236 ± 2.5 | 2.347 ± 0.58 |
| random_search | 4.689 ± 3.4 | 1.684 ± 0.78 |

**The control and the agent have separated, and that is the finding.** In Phase 3
they were identical — both landed exactly on Branin's optimum, because both were
reciting it. Here `single_shot_llm`, which plans blind, is the *worst* LLM row
and loses to random search on branin_i1; `agent_no_reflection`, which sees
results, is eleven times better than it. The gap between them is the loop, and
in Phase 3 that gap was invisible because memorisation had saturated both.

Read directly from the proposals. On the original Branin the blind plan opened
with (−π, 12.275), (π, 2.275), (3π, 2.475) — the three published minima to four
decimal places. On `branin_i1` it opens:

```
(0.0, 0.0)  (0.0, 1.0)  (1.0, 0.0)  (1.0, 1.0)   <- corners
(0.25, 0.25)  (0.5, 0.5)  (0.75, 0.75)           <- centre and diagonal
```

A textbook space-filling design. The model stopped recalling and started
searching.

### The live memorisation check, and an honest correction

The marked live test asks the same strategy for a blind plan on the original
benchmark and on the instance, and scores the best proposal in each:

| Problem | Blind first-batch regret |
|---|---|
| `branin` (original, anonymised text, true bounds) | 1.545 |
| `branin_i1` (shifted, rotated) | 3.035 |

**This is much weaker evidence than Phase 3's, and the difference matters.**
Phase 3's number on the original Branin was 3.578×10⁻⁷ — the global optimum, to
four decimal places, blind. That was **Gemini**. `qwen3:4b-q4_K_M` reaches 1.545:
better on the original than on the instance, but nowhere near recall.

So the correct statement is narrower than "the model had memorised the
benchmarks". A frontier hosted model had. **The local 4B model does not appear
to have memorised Branin's optima**, and the 2× gap here is too small, on one
comparison, to attribute to memory rather than to the original's domain simply
being a different problem.

That does not weaken the fix — it makes it *cheap insurance that is now
permanent*. The instances are unrecoverable from the prompt by construction, and
that property holds whatever model is plugged in later. Phase 4's paid frontier
comparison, which §5 requires, would otherwise have walked straight back into
Phase 3's trap.

### How much of that survives the noise floor

`phase35-noise`: the same baselines at 30 seeds, resampled 5 at a time — how far
apart two 5-seed means can drift when nothing differs but the seed.

| Strategy | Benchmark | Mean regret | 90% interval of a 5-seed mean | Spread |
|---|---|---|---|---|
| optuna_tpe | branin_i1 | 4.016 | [1.855, 6.820] | **4.965** |
| random_search | branin_i1 | 7.071 | [3.666, 10.95] | **7.279** |
| optuna_tpe | hartmann6_i1 | 1.686 | [1.297, 2.058] | 0.762 |
| random_search | hartmann6_i1 | 2.100 | [1.683, 2.457] | 0.774 |

This is the number that stops the headline being overclaimed, and the two
benchmarks come out differently.

**On branin_i1 the agent's advantage is real, but not because of the means.**
The mean gap to TPE is 2.46, comfortably *inside* TPE's own 4.965 spread — so
the mean comparison on its own proves nothing. What does hold up is the rank
test on paired seeds: the agent's five runs land at 0.069, 0.278, 0.386, 0.507,
0.665 and TPE's at 0.632, 1.896, 2.170, 2.341, 7.151. Only one of twenty-five
pairs inverts, giving U=1 and an **exact two-sided p of 1/63 = 0.0159**. The
agent is not so much *better on average* as dramatically more **consistent** —
TPE at 20 evaluations on this instance is erratic, which is what the enormous
noise floor is describing.

A rank test rather than a t-test, deliberately: these regret distributions are
heavily right-skewed (TPE's worst seed is 7.15 against a median of 2.17), which
breaks the normality a t-test assumes and lets one unlucky seed drive the
result. `method="exact"` is now named rather than left to scipy's `auto`; at
n=5 `auto` already picks exact, but the normal approximation would report 0.0216
for this same comparison and a later experiment with more seeds would have
switched to it silently.

**On hartmann6_i1 the advantage does not clear the floor.** Mean gap to TPE is
0.49 against a spread of 0.762, and p=0.151. **No claim is made here.** The
agent leads, and that is all the data supports.

The one comparison that is unambiguous on both benchmarks is
`agent_no_reflection` against `single_shot_llm` — p=0.0119 and p=0.0236 — which
is the "does looping help at all?" question the control exists to answer. On
these instances, it does.

### Was TPE benchmarked in its worst regime? Yes — and fixing it makes it worse

Optuna's `TPESampler` was constructed with a seed and nothing else, so
`n_startup_trials` was its default **10**. At a 20-evaluation budget that means
**half of TPE's run is random sampling** before the model takes over. A baseline
crippled by an unexamined default is not a baseline, and this one had gone
unexamined through three phases.

The obvious fix is to lower it. I measured before doing so — median final regret
over 40 seeds, varying only that value:

| budget | | start=3 | start=5 | start=8 | start=10 |
|---|---|---|---|---|---|
| 20 | branin_i1 | 2.723 | 2.664 | **1.686** | 1.689 |
| 20 | hartmann6_i1 | 1.646 | **1.530** | 1.639 | 1.709 |
| 40 | branin_i1 | 0.534 | 0.679 | 0.576 | **0.362** |
| 80 | branin_i1 | **0.094** | 0.127 | 0.136 | 0.122 |

**Lowering it does not help and mostly hurts.** The estimator needs those
observations to build a density over; starved of them it models the space badly
and the "wasted" random trials turn out to be the price of the model working at
all. At budget 40 the default is clearly best.

So the honest response to "TPE is in its worst regime" is: *yes, and that is
inherent to the method at this budget, not an artefact of a badly chosen
constant.* The value is now a named constant carrying that table as its
justification, and the report prints it in the metadata with the warning
attached, so no reader has to wonder.

This is the second time this phase that a confident mechanical intuition — "half
the budget is wasted", "constrained decoding is the real risk at 4B" — did not
survive being measured. Both are recorded because the measurement is the
contribution, not the guess.

### Rescoping the claim: sample efficiency, not superiority

Revised after Phase 3.5 shipped, because the original wording invited a reading
the data does not support.

"The agent reaches regret 0.381 where TPE reaches 2.838" is true and, stated
without its budget, misleading. TPE on the *original* Branin at 200 evaluations
reaches **0.0188** — an order of magnitude better than anything the agent has been
measured at. Bayesian optimisation spends its early evaluations building a
surrogate and only then exploits it; scoring it at 20 evaluations scores it
before it has begun. Two specifics make that concrete:

- Optuna's `n_startup_trials` is **10**. At a 20-evaluation budget, **half of
  TPE's run is random sampling** by construction.
- Phase 3 already found the same shape from the other direction: TPE's 5.7×
  advantage over random search on Hartmann-6 at 200 evaluations *vanished* at 50.

So the finding is **sample efficiency under a tight budget**: when evaluations
are the scarce resource, a planner that can read results and reason about them in
language gets useful faster than a surrogate that must first be fitted. That is a
narrower claim and the one worth making, because it is the regime a real
simulator lives in — the whole premise of the project is that each evaluation
costs minutes.

What was changed: the headline in `README.md`, the framing above this section,
and the interview angle. What was *not* changed: any number. The measurements
stand; only the sentence wrapped around them was too wide.

### The budget sweep, and where the lead changes hands

`sil_agent/eval/sweep.py` plus a `sweep` CLI command. Each point is an
**independent run at that budget**, never a prefix of a longer one — the planner
is told `max_evaluations` in its prompt, so a run that knows it has 80
evaluations explores differently from one that knows it has 10, and truncating
the long run would measure the wrong thing. (For TPE the distinction does not
matter; for the agent it does, so the sweep pays for it on both.)

Median final regret on **branin_i1**, with a rank test at each budget:

| Budget | agent | TPE | leading median | p (exact) | |
|---|---|---|---|---|---|
| 10 | 3.02 | 10.98 | agent | 0.0556 | not significant |
| 20 | **0.386** | 2.17 | agent | **0.0159** | **significant** |
| 40 | 0.805 | 0.632 | TPE | 0.6905 | not significant |
| 80 | 0.164 | 0.191 | agent | 1.0000 | not significant (n=3) |

**The medians cross near 37 evaluations, and that crossing is not a finding.**
This is the part I nearly got wrong. The curves cross, the interpolation is
sound, and it would have made a satisfying headline — "TPE overtakes at 37" —
except that *no budget above the crossing shows a significant difference*. At 40
the two distributions are almost entirely overlapping (p=0.69); at 80 they are
indistinguishable (p=1.00). The medians swap places twice.

What the sweep actually shows is narrower and duller than a crossover: **the
agent's advantage is real at 20 evaluations and has dissolved by 40.** It does
not reverse — TPE never demonstrably overtakes within the budgets measured — it
simply stops being detectable. The report now leads with the per-budget test and
prints the crossing only as a median crossing, with that caveat attached.

Having spent this session removing one overclaim, replacing it with a
better-dressed one would have been the easy mistake. The per-budget rank test is
in `sweep.py` for that reason: the medians alone invite exactly that reading.

**hartmann6_i1**

| Strategy | 10 evals | 20 evals | 40 evals |
|---|---|---|---|
| agent_no_reflection | **1.323** | **0.886** | **0.710** |
| optuna_tpe | 2.54 | 1.223 | 1.052 |

The agent leads at all three, and again this is weaker than it looks: the
Phase 3.5 noise floor on this instance is 0.762, wider than any of these gaps.
The agent's budget-80 cell was not run — see below.

**Scope actually delivered.** The 80-evaluation cell on branin_i1 has three seeds
rather than five, and hartmann6_i1 has no 80-evaluation agent cell at all. Each
80-evaluation run is 80 sequential model calls and the harness running them kept
being cut short; five re-issues got branin to three seeds. Rather than quietly
plot a three-seed median next to five-seed ones, `sweep.py` detects unequal cells
and the report prints a warning naming them. Stated here as an incomplete sweep,
because the alternative — dropping the point and showing a clean three-budget
curve — would have looked more finished and told the reader less.

### Structured output

| Strategy | Model calls | Valid first try | Needed repair | Never valid | Compliance |
|---|---|---|---|---|---|
| agent_no_reflection | 200 | 200 | 0 | 0 | **100.0%** |
| single_shot_llm | 10 | 10 | 0 | 0 | **100.0%** |

210 calls, not one repair. That number is only meaningful next to the same
measurement with the grammar switched off — see the ablation below — because
100% is what constrained decoding is *supposed* to produce, and a metric that
cannot fail is not a measurement.

### What the 4B model is bad at

Not JSON — the grammar handles that. It is bad at **producing a plan of the
requested length**. `single_shot_llm` was asked for 20 proposals and returned
fewer on most runs: mean 16 evaluations on hartmann6_i1 against a 20 budget, and
one run that yielded only 4. Those runs terminate `EXHAUSTED`, which the harness
already reports honestly rather than padding.

Grid search is worse still, for the reason Phase 2 documented: at 20 evaluations
in 6-D it manages one point per axis, so it evaluates a single point and stops.

### Constrained decoding did not earn its place — a negative result

`TECHNICAL_DESIGN.md` §5 predicts structured output as *"the real risk at 4B"*
and requires the grammar rather than prompting. 100% compliance in the table
above is consistent with that — and is also exactly what a grammar produces by
construction, so on its own it proves nothing. A metric that cannot fail is not
a measurement.

So it was switched off. The 200 planner prompts the experiment actually sent
were replayed from the `llm_calls` audit trail, 40 sampled, at the temperature
the experiment ran at:

| Condition | Schema-valid on first attempt |
|---|---|
| `format` present (grammar-constrained) | 40 / 40 = **100%** |
| `format` absent (asked in the prompt only) | 40 / 40 = **100%** |

**`qwen3:4b-q4_K_M` returns schema-valid JSON for these prompts whether or not
it is constrained.** The predicted risk did not materialise at this model size
on this workload.

The grammar stays, for a reason that survives the result: it converts an
*observation* into a *guarantee*, and it costs nothing. But the design document's
claim should be read as unproven rather than confirmed, and the honest headline
is that the compliance number is high because the model is adequate, not because
the grammar rescued it.

Worth noting what the first version of this ablation got wrong, because it
nearly went in the log: it ran at temperature 0.0 while the experiment ran at
0.7, and a bug meant its "with history" prompts were actually empty. It reported
the same 100%/100% for weaker reasons. Replaying the real recorded prompts is
what makes the number mean anything — which is the `llm_calls` table doing the
job it was built for in Phase 3.

### The convergence charts were wrong, and the report said so a phase early

Phase 3's log flagged that the palette was CVD-validated for three series and
that Phase 4 would need more. Phase 3.5 needed **five**, and `colour_for` cycled
the palette with `% len(...)` — so `agent_no_reflection` and `random_search` were
painted the same blue, and `grid_search` and `single_shot_llm` the same orange.
Two lines meaning different things in one colour is worse than no chart.

The palette is now six validated slots (worst adjacent CVD ΔE 9.2 deutan / 7.8
tritan, worst normal-vision ΔE 22.6, all inside the lightness band and above the
chroma floor), plus per-series line styles as a redundant channel — the tritan
figure sits in the band that is only legal alongside a secondary encoding. Past
six slots `colour_for` now **raises** rather than wrapping, so the next person to
add a strategy is made to choose small multiples deliberately instead of
discovering two identical blue lines.

Caught by rendering the PNG and looking at it. The code was clean and the
validator had passed for the palette it was given; nothing but looking would have
found it. That is the second time in three phases a chart defect survived
correct-looking code.


## Interview angle

I built a benchmark that made my own agent look brilliant, and caught it before
it reached a report. The agent scored near-zero regret in five evaluations on a
function it had never seen — not because it reasoned, but because the randomly
placed optimum landed at (0.6001, 0.4017) and the model proposes round numbers.
Language models do; random search and TPE do not, so the benchmark was
differentially easy for exactly the thing I was measuring. The fix was to reject
optima that sit near the grid, and the evidence is a test that sweeps every round
number and asserts real regret survives.

That is the answer to "tell me about a time you found a bug in your own
methodology." The supporting story is that the same phase caught a second one:
five seeds of an LLM strategy returning byte-identical results, because greedy
decoding ignores the seed — five copies of one measurement reported as five
replicates, which is the same pseudoreplication I had already found in grid
search a phase earlier.

The result underneath is worth stating too, and stating at the right width. On a
function the model demonstrably cannot have read about, a local 4-billion-parameter
model with a feedback loop reaches regret 0.38 at twenty evaluations where
Optuna's TPE reaches 2.84 — and the same model with the loop removed reaches
4.24, worse than random search. The gap between those two is the loop, and in
Phase 3 it had been invisible because memorisation saturated both.

**That is a claim about sample efficiency under a tight budget, not about being a
better optimiser**, and the budget sweep is what keeps it honest. Running both at
10, 20, 40 and 80 evaluations, the agent's advantage is statistically supported
at 20 (p=0.016) and gone by 40 (p=0.69) — the medians cross near 37, but no
budget above the crossing shows a real difference, so the right description is
*the edge dissolves*, not *TPE overtakes*. I nearly reported the crossover as the
result; it was a better story and the per-budget rank test says it is not one.

That is the honest version of the finding, and it is still the useful one: it
says use this when evaluations cost minutes, and stop expecting anything from it
once they do not. I also report where the claim stops entirely: on the
six-dimensional instance the lead sits inside the measured seed noise floor, so I
do not make it.
