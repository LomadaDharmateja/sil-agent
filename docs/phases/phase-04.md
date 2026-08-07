# Phase 4 — the critic and the replanner

Built 2026-08-07. The experiment the project exists to run.

**The answer is no.** On these two benchmarks, at twenty evaluations, with a
local 4B model, reflection did not pay. The reflecting agent's mean regret on
`branin_i1` is 7.10 against the planner-only control's 0.33, for 3.8× the
tokens — though a third arm shows that about half of that degradation comes from
the reworded prompt rather than from reflection itself, and no single step
clears significance at five seeds.

That is the result, it was measured honestly, and this log is mostly about how it
was measured and what else fell over on the way.

## 1. Why this phase exists

`TECHNICAL_DESIGN.md` §13 calls Phase 4 *"the money experiment: does reflection
pay?"*. Phases 1–2 built the measurement harness before there was an agent so
that this question could be asked properly; Phase 3 built the loop; Phase 3.5
existed solely to make the question *answerable* — on functions the model cannot
have read about, with a model that will still exist in two years.

The question, stated so it could come out either way:

> Does an agent that diagnoses its own results and re-plans from that diagnosis
> find better designs, **at an equal number of simulator calls**, than the same
> agent with reflection removed?

`agent_no_reflection` was the control and already existed. `agent_full` is the
treatment. Everything else — simulators, seeds, evaluation budget, guard, model,
harness — is held constant.

## 2. What I built

| File | Responsibility |
|---|---|
| `agent/critic.py` | The critic. Explains a result it is *told* the grade of. |
| `agent/replanner.py` | The replanner. A named next mode over the critic's output. |
| `prompts/critic.v1.md` | Critic template. |
| `prompts/replanner.v1.md` | Replanner template. |
| `prompts/planner.v2.md` | Planner + reflection block. **v1 untouched.** |
| `tests/test_reflection.py` | 26 tests: Rule 2, failure paths, the feedback loop. |
| `tests/test_determinism.py` | Two `live` tests for the reproducibility claim. |

Changed: `strategies/base.py` (`Reflects`, `Reflection`), `agent/loop.py` (calls
the reflector, obeys none of it), `strategies/llm_agent.py` (`AgentFull`,
`AgentPromptControl`), `strategies/registry.py`, `agent/stagnation.py` (the two
detectors that needed a critic), `agent/state.py` (`CostRecord.plus`, per-request
accounting), `eval/harness.py` (run locking), `eval/metrics.py` and
`eval/report.py` (reflection and cost-of-reflection tables), `cli.py`.

## 3. How it works

### The critic runs in the loop, and the loop does not know what a critic is

`TECHNICAL_DESIGN.md` §3 puts `critic.evaluate` after the simulator, which is
also the only place it can go: `Strategy.propose` returns *before* the simulator
runs. But the same loop runs random search, so the loop must not grow a
dependency on critics.

The existing answer to that shape of problem is `ReportsCost` — an optional
protocol the loop checks at runtime, so baselines do not implement an accounting
method to report zero. Reflection got the same treatment:

```python
if isinstance(strategy, Reflects):
    reflection = strategy.reflect(goal, history, candidate, outcome, computed, best)
```

A strategy that does not implement it stores the computed evaluation and a
placeholder decision — exactly what every strategy did through Phase 3.5.
`build_strategy(name)` stays the single source of truth for what a strategy is,
so there is no second switch anywhere saying "and this one also reflects". That
is how a run gets labelled `agent_full` while quietly not reflecting.

### Rule 2 is enforced by the type, not by the prompt

This is the design decision I care most about in the phase.

`Evaluation` has six fields: three computed from the simulator (`improved`,
`delta_vs_best`, `feasible`) and three from the model. The obvious build asks the
model for an `Evaluation` and trusts it to leave the first three alone.

Instead the critic's schema **does not contain them**:

```python
class CriticVerdict(BaseModel):
    diagnosis:   str
    hypotheses:  list[str]
    confidence:  float
```

The loop assembles the stored `Evaluation` itself, taking the computed three from
the oracle and the prose three from the verdict, in one function
(`evaluation_from`). The model is *shown* `improved` and `delta_vs_best` in its
prompt as facts to explain, and has no channel through which to return a
different value.

A convention the prompt asks for can be ignored. A field that does not exist
cannot be filled in. `test_the_critic_cannot_return_a_computed_field` asserts the
field set, so the guarantee cannot be deleted silently.

### Reflection reaches the next proposal through the database

`Episode.evaluation` and `Episode.decision` have been columns since Phase 1 —
the schema was fixed then *specifically* so Phase 4 would need no migration.
`describe_reflection(history)` reads the previous episode's diagnosis and
decision out of `history` and renders them into the planner prompt.

Nothing is held on the strategy object, so a run killed after episode 7 and
resumed rebuilds episode 8's prompt, reflection included, from the episodes
table. Rule 1 survives untouched.

Worked example, from the first smoke run — episode 1's critic said *"increasing
p1 from 0.3 to 0.5 while keeping p2 at 0.7 would decrease the objective"*, and
episode 2's planner rationale opened *"the previous result at p1=0.3, p2=0.7 had
a high objective value, suggesting that increasing p1 towards 0.5…"*. The loop
closes.

### `TERMINATE` is recorded and not obeyed

The replanner may recommend stopping. The loop does not stop, behind a
`honour_terminate: bool = False` that stayed off for the whole experiment.

Two reasons, and the second is operative: Rule 2 says deterministic code decides
termination; and a strategy that talks itself into quitting at evaluation six has
not lost the same contest the others were in, it has set its own budget. The
evaluation budget being identical across strategies is the single property the
comparison rests on.

It was recommended **once in 200 episodes**, so nothing turned on it — but the
number is in the report, which is strictly better than a confound.

The schema offers four actions, not `ReplanAction`'s six. `DECOMPOSE` and
`ESCALATE` have nothing behind them (Phase 11), and a model offered an action
nothing will act on picks it eventually.

### Locking, finally wired

Phase 3 built `services/locks.py` for exactly the collision Phase 3.5 then hit.
`execute_cell` now wraps `run_loop` in `build_lock(cell.run_id)`; `LockUnavailable`
marks the cell `"locked"` and moves on rather than failing the matrix; and
`run_loop` gained a `heartbeat` callback, called after each *durably written*
episode, so a lost lease stops the worker immediately with the work committed.

The natural key on `(run_id, idx)` stays and stays load-bearing: a lock has a TTL
and a stalled worker can outlive its lease, whereas a unique index cannot be
outlived. **The lock removes the waste; the key removes the corruption.**

It earned its place within an hour — see §5.

## 4. Key decisions and trade-offs

**`planner.v2` rather than editing v1.** `agent_full` needs a reflection block.
Editing `planner.v1.md` would have changed `call_key` for every recorded Phase 3.5
call, making `phase35-main` unreplayable, *and* silently altered
`agent_no_reflection` — the control — mid-experiment. So v2 is v1 plus a block,
v1 is untouched, and `render_planner_prompt` always passes `reflection_block`
because `string.Template.substitute` ignores values a template does not mention.
A test asserts v1 renders identically with and without it.

**The confound, and the arm that controls it.** `agent_full` differs from
`agent_no_reflection` in two ways: reflection content, and prompt wording. A
difference could be either. `agent_prompt_control` holds the prompt at v2 and
removes only the content. It runs as its own experiment because `report.py`'s
validated colour palette has six slots and *raises* on a seventh rather than
reusing a hue — a guard Phase 3.5 added after two lines came out the same blue.
Working around it by adding a hue would have been the wrong response to it.

**The detectors are implemented, tested and switched off.** A detector firing at
evaluation 12 gives one arm twelve evaluations against everyone else's twenty,
which is the fairness problem above arriving by a different door. `run_loop`
takes `detectors=None` by default and every matrix since Phase 2 has run that
way. See §5 for what they would have caught, which is nothing.

**Cost is reported next to regret.** A reflecting episode is three model
requests against one, for the same single simulator call. Holding the
*evaluation* budget equal is right for a project premised on a simulator call
costing minutes — but it makes any win a claim about sample efficiency, not
efficiency. Phase 3.5 had to retrofit that rescoping after publishing; this time
the table was built before the numbers existed.

**What would have been easier but worse:** asking the model for a full
`Evaluation` and validating that it had not changed the computed fields. It
works, it is one fewer function, and it makes Rule 2 a runtime check on a value
the model was invited to send — instead of a fact about the schema.

## 5. What went wrong

**The headline is a negative result.** Reflection did not help. On `branin_i1`
the reflecting arm was worse than the control on every summary statistic; on
`hartmann6_i1` there is no difference worth reporting. The brief predicted this
as expected-failure #2 and #6, which is the only reason it is not a surprise.

**And the confound arm changed what the negative result means.** The brief
flagged that `agent_full` differs from the control in two ways — reflection
content *and* prompt wording — and specified a third arm to separate them. I had
already written "reflection made it worse" in my notes before that arm finished.
It ran, and about half the `branin_i1` degradation turns out to be present in the
reworded prompt with no reflection content at all (mean 0.33 → 1.67 → 7.10). The
two-arm comparison would have attributed all of it to reflection. That arm cost
200 model calls and was the highest-value thing in the phase.

**I nearly reported a reproducibility success that was a coincidence.** The
phase re-runs `agent_no_reflection` under a new experiment name, which gets new
`run_id`s, misses the replay cache, and re-issues every call at the same seed
against the same pinned model — a free test of Phase 3.5's central claim.

The first two cells matched Phase 3.5's regret *to sixteen digits* and I wrote
that down as reproduction confirmed. It was not. Pulling the trajectories apart:

```
seed 1  phase35: (0.5,0.5) (0.3,0.7) (0.7,0.3) (0.6,0.4) ...  best (0.65,0.35)
seed 1  phase4 : (0.5,0.5) (0.0,0.0) (0.5,0.0) (0.7,0.3) ...  best (0.65,0.35)
```

Different runs from episode 1 onward, landing on the same best point at different
episodes (5 and 13). They agree because the model proposes **round numbers**, so
independent trajectories converge on the same grid point — which is Phase 3.5's
own round-number finding arriving from a new direction. Seed 2 genuinely did
reproduce, and the tell is that its best sits at `(0.6476270427544129,
0.3813805360605672)`: a non-round value cannot coincide.

Measured across all ten cells:

| | |
|---|---|
| Exact trajectory reproduction | **1 / 10** |
| Same final best value | 3 / 10 |

**So final-score agreement overstates reproduction threefold, and
`TECHNICAL_DESIGN.md` §5's headline argument for going local does not hold on
this hardware.** The Phase 3.5 log states "Two runs with the same seed match:
**yes**" in a table of properties. That row was asserted, not measured. This is
the correction.

What survives, and it matters: **replay is exact, re-execution is not.** A run
replayed from the `llm_calls` table reproduces byte-for-byte, because it reads
stored text. The reproducibility this project actually has comes from the audit
trail built in Phase 3, not from pinning the model. That is a narrower claim and
a true one.

My first explanation was that the model does not fit the 4 GB card — inference
runs 36%/64% CPU/GPU — so reduction order across the split drifts between loads.
That was a guess, and this project has a rule about those, so I measured it.
`tests/test_determinism.py` issues one short request twice in-process, and again
from a fresh interpreter:

| Condition | Reproduced? |
|---|---|
| Same seed, twice in one process | **yes** |
| Same seed, separate process, model reloaded | **yes** |

**Both pass.** The seed is genuinely wired, and an isolated request is
deterministic across a reload on exactly the hardware that produced 1-in-10
above. So the simple explanation is wrong, and the two obvious suspects — an
unseeded sampler, and a hardware reload — are both ruled out.

What is left is *sequence*. A cell is twenty sequential calls sharing long
prefixes, issued into a server session that had already processed hundreds of
requests from earlier cells. Ollama reuses KV cache across requests, so how much
of a prompt is recomputed rather than reused depends on what preceded it —
which makes a generation a function of the request *and its history on that
server*, not the request alone. Two runs that reach a byte-identical prompt by
different routes need not produce identical output from it, and the first
divergence in eight of ten cells was at episode 1, 2 or 3, before histories had
had time to differ much.

Stated as the surviving hypothesis rather than a result: I have ruled out two
explanations and not confirmed a third. What is *measured* is the 1-in-10, and
that is enough to retire the design claim.

**The critic's confidence is a constant.** Across 200 reflected episodes:

| Confidence | Episodes |
|---|---|
| 0.60 | **175** |
| 0.65 | 11 |
| 0.40 | 8 |
| 0.30 | 5 |
| 0.70 | 1 |

`qwen3:4b-q4_K_M` reports 0.60 for seven episodes in eight, whatever happened.
`ConfidenceDecline` — a detector this phase built, specified by
`TECHNICAL_DESIGN.md` §3 — is therefore reading a constant and can never fire.
The detector is correct; the signal is not there. A field being schema-valid says
nothing about it carrying information, and this is the clearest example of that
in the project so far.

**No detector would have caught the run that failed worst.** `agent_full` on
`branin_i1` seed 2 finished at regret 19.5 by doing this:

```
ep 4 (0.500,0.750) obj=30.012 EXPLOIT     ep 9 (0.500,0.975) obj=22.179 EXPLOIT
ep 5 (0.500,0.800) obj=28.209 EXPLOIT     ep10 (0.500,0.990) obj=21.679 EXPLOIT
ep 6 (0.500,0.850) obj=26.447 EXPLOIT     ep11 (0.500,0.995) obj=21.513 EXPLOIT
ep 7 (0.500,0.900) obj=24.718 EXPLOIT     ep12 (0.553,1.000) obj=20.174 EXPLOIT
ep 8 (0.500,0.950) obj=23.018 EXPLOIT     ep13 (0.553,1.000) obj=20.173 EXPLOIT
```

Twelve consecutive evaluations micro-stepping `p2` toward a boundary along a
fixed `p1=0.5`, in a region an order of magnitude away from the optimum at
`(0.65, 0.35)`. Every diagnosis was locally *true* — raising `p2` did reduce the
objective each time — and each confirmed EXPLOIT, so the loop followed a local
gradient off a cliff. This is the failure the brief called "a confident wrong
diagnosis biases the next proposal", except the diagnoses were not even wrong.

I replayed all three detectors over every prefix of all ten `agent_full` runs:
**none of them would have fired on any run.** `DiversityCollapse` has
`epsilon=0.02` and those steps are 0.05 wide in normalised space, so the
pathology it exists to catch sat just outside its threshold. The windows were
tuned in Phase 3 against a hypothetical; this is the first real stagnation the
project has produced and the tuning does not cover it.

**`RepeatedDiagnosis` failed its own test case.** The first version compared word
sets and two diagnoses differing only in *"suggesting"* against *"suggests"*
scored 0.71 against a 0.75 threshold. Inflection is the same thought in different
words — precisely what the detector claims to see through — so it was measuring
grammar rather than content. Fixed with a four-suffix stemmer. Caught because the
test case was written from what a model actually produces rather than from what
would pass.

**The compliance metric became wrong the moment an episode stopped being one
call.** `model_calls` counted *episodes* with `cost.calls > 0`, which was right
while every episode was one model call and silently wrong at three — it would
have divided the denominator by three and inflated any compliance rate computed
from it. Fixed by counting logical requests (`CostRecord.requests`), with a
fallback so episodes written before the field existed still report their original
numbers. A derived number that was correct by accident is the kind that stays
wrong for a long time.

**Phase 3.5's log was wrong about where locking was wired.** It says *"locking is
wired for the CLI's single-run path, not for `execute_cell`"*. In fact
`build_lock` had **zero** production callers: it was written in Phase 3, tested
against a fake Redis, and never called from anywhere. Both paths were wired this
phase. Worth recording because the earlier log made the gap sound smaller than it
was, and I only found out by grepping for callers rather than trusting it.

**The matrix was killed twice and resumed itself.** Same as Phase 3.5: the runs
outlive the harness that launches them. Re-issuing the identical command skipped
the 39 finished cells and continued — no resume flag, no job table. On the first
re-issue one cell printed:

```
[ 40/60] single_shot_llm/hartmann6_i1/seed=5   locked by another worker - skipped
```

The killed worker had died holding its lease, which had not yet expired. **The
lock built this phase did its job within an hour of being wired**, and did it
better than Phase 3.5's outcome — there the collision was caught by the natural
key only *after* the loser had paid for a simulation and a model call. The cell
was picked up on a later invocation once the 120-second TTL lapsed. I also
checked for a surviving child process first, which is the operational lesson
Phase 3.5 wrote down.

## 6. What this unlocks

Phase 5 (episodic memory) now has a control worth beating and a diagnosis of
*why* the obvious thing failed. The failure is not that the critic produced
nonsense — it produced locally accurate analysis with 100% schema compliance and
zero infrastructure failures. The failure is that **local accuracy plus EXPLOIT
is a hill-climber**, and a hill-climber from a bad start at twenty evaluations
loses to space-filling.

That points somewhere specific rather than "try harder":

- The replanner chose EXPLOIT 109 times of 200 and EXPLORE 70. The prompt tells
  it to prefer EXPLORE early, and it does not. Prompt A/B testing (Phase 11) has
  a concrete first hypothesis, and the harness to test it.
- The prompt control makes that hypothesis sharper: v2's directive sixth rule
  degrades `branin_i1` on its own, before any reflection exists. The first A/B to
  run is v2 with that rule softened from "follow it unless the results plainly
  contradict it" to something advisory.
- `DiversityCollapse` needs re-tuning against a real trajectory, now that one
  exists. `epsilon=0.02` is too tight by roughly a factor of three.
- `confidence` is inert at 4B and should not be built on until a stronger model
  is measured against it — which is the paid frontier comparison §5 asks for, and
  now has a specific question to answer rather than a formality.
- The budget question is open in the right way: Phase 3.5's sweep exists, and
  "at which budgets does reflection pay?" is answerable by adding `agent_full`
  cells at 40 and 80 to the same axes.

## Numbers

| Measurement | Value |
|---|---|
| Tests | **357** — 351 passing offline, 2 skipped in high dimension, 4 live and deselected |
| Tests requiring a network or API key | **0** |
| Suite runtime | ~24 s offline (36 s with the GPU busy alongside) |
| Model | `qwen3:4b-q4_K_M`, pinned including quantisation |
| GPU residency | 3.6 GB, **36% / 64% CPU/GPU**, `num_ctx` 4096 |
| Model calls in `phase4-main` | **810** (600 agent_full + 200 agent_no_reflection + 10 single-shot) |
| Model calls in `phase4-control` | 200 |
| Schema compliance | **100%** (810/810 first try, zero repairs) |
| Reflection failures | **0** |
| Exact trajectory reproduction, re-executed at the same seed | **1 / 10** |
| Isolated request reproduced across a process restart | yes (measured) |
| Worst-case prompt | 1,051 tokens of 4,096 (replanner, 6-D, full history) |
| LLM spend | **€0.00** |

### The headline — final regret, mean ± sd over 5 seeds, 20 evaluations

| Strategy | branin_i1 | hartmann6_i1 |
|---|---|---|
| **agent_no_reflection** | **0.331 ± 0.19** | **0.944 ± 0.43** |
| agent_full | 7.104 ± 9.2 | 1.048 ± 0.56 |
| grid_search | 2.196 ± 0 | 1.956 ± 0 |
| optuna_tpe | 2.838 ± 2.5 | 1.540 ± 0.64 |
| single_shot_llm | 2.849 ± 0.41 | 1.956 ± 0 |
| random_search | 4.689 ± 3.4 | 1.684 ± 0.78 |

### Does reflection pay? No.

| Benchmark | median `agent_full` | median control | p (exact, two-sided) | |
|---|---|---|---|---|
| branin_i1 | 0.507 | 0.386 | 0.0556 | not significant |
| hartmann6_i1 | 0.729 | 0.714 | 0.8413 | not significant |

**Neither difference clears significance, so the honest statement is that no
effect was demonstrated in either direction.** But the mean on `branin_i1` tells
a second story the median hides — 7.104 against 0.331, with a standard deviation
of 9.2:

| seed | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `agent_full` | 0.507 | **19.50** | **14.61** | 0.507 | 0.395 |
| `agent_no_reflection` | 0.507 | 0.386 | 0.497 | 0.196 | 0.069 |

Three of five seeds are fine and two are catastrophic. **Reflection did not lower
the typical result so much as add a failure mode.** Phase 3.5's finding about
this agent was that it beat TPE by being dramatically *more consistent* than it;
adding reflection destroyed exactly that property. The control's five runs span
0.07–0.51; the treatment's span 0.40–19.5.

That is why the mean and the rank test disagree, and why both are reported. The
rank test says "no demonstrated difference in the typical case" and is right. The
spread says "this version can fail badly and the other one cannot", which at n=5
is an observation rather than a claim, and is the one I would act on.

### Was it the reflection, or the prompt that carries it? Partly the prompt.

`agent_full` differs from the control in *two* ways — reflection content, and
`planner.v2`'s wording. `agent_prompt_control` (experiment `phase4-control`)
holds the prompt at v2 and removes only the content, which splits the difference
into two steps:

**branin_i1**

| Arm | Prompt | Reflection | mean | sd | median |
|---|---|---|---|---|---|
| `agent_no_reflection` | v1 | no | **0.331** | 0.19 | 0.386 |
| `agent_prompt_control` | v2 | no | 1.666 | 2.49 | 0.507 |
| `agent_full` | v2 | yes | 7.104 | 9.25 | 0.507 |

**hartmann6_i1**

| Arm | mean | sd | median |
|---|---|---|---|
| `agent_no_reflection` | 0.944 | 0.43 | 0.714 |
| `agent_prompt_control` | 0.966 | 0.63 | 0.820 |
| `agent_full` | 1.048 | 0.56 | 0.729 |

| Step | branin_i1 | hartmann6_i1 |
|---|---|---|
| Prompt effect (v1 → v2, both unreflected) | p=0.1508 | p=0.8413 |
| Reflection effect (v2 → v2 + reflection) | p=0.6905 | p=1.0000 |

**Neither step is significant at n=5, and the arm changed the conclusion
anyway.** On `branin_i1`, about half the degradation in the mean is already
present *before any reflection content exists* — the reworded prompt alone takes
the mean from 0.33 to 1.67. Without this arm I would have attributed all of it to
reflection, which is what the two-arm comparison invites and what I had written
down before running it.

The robust signal across all three arms is not the mean but the **spread**, which
escalates monotonically at both steps: sd 0.19 → 2.49 → 9.25. Both changes add
variance, and reflection adds more of it than the rewording does. On
`hartmann6_i1` nothing moves at all.

The plausible mechanism is that v2's sixth rule — *"the review section states a
direction […] follow it unless the results plainly contradict it"* — is
directive, and a 4B model reads directive instructions literally. It commits to a
mode even when the review block, in the control, explicitly says no review is
available. Testing that is prompt A/B work, which is Phase 11 and now has its
first concrete hypothesis.

### What reflection cost

| Strategy | Model requests | Requests per evaluation | Prompt tokens | Completion tokens |
|---|---|---|---|---|
| agent_full | 600 | 3.00 | 673,036 | 65,901 |
| agent_no_reflection | 200 | 1.00 | 176,354 | 23,809 |

**3.8× the prompt tokens and 3× the calls, for a worse result.** At an equal
evaluation budget this is invisible in the headline table, which is why the table
exists.

### What the replanner decided

| EXPLOIT | EXPLORE | REPAIR | TERMINATE |
|---|---|---|---|
| 109 | 70 | 20 | 1 |

Fifty-five per cent EXPLOIT across the whole run, including early episodes where
the prompt explicitly says exploration is almost always right. Combined with the
seed-2 trace above, this is the mechanism behind the result.

## Interview angle

I built the reflection loop my design document called "the money experiment",
measured it against its own control at five seeds on two benchmarks the model
cannot have memorised, and found that **it did not pay** — the planner-only agent
reaches regret 0.33 where the reflecting agent reaches 7.10, for 3.8× the tokens.
I shipped that result rather than tuning until it inverted.

The interesting part is the diagnosis. The critic was not broken: 100% schema
compliance over 600 calls, zero failures, and diagnoses that were *locally
accurate every time*. On the worst run it correctly observed that raising one
parameter reduced the objective, twelve times in a row, while the search
micro-stepped toward a boundary an order of magnitude away from the optimum.
Local accuracy plus a replanner that chose EXPLOIT 55% of the time is a
hill-climber, and at twenty evaluations a hill-climber from a bad start loses to
space-filling. Reflection did not lower the typical result; it added a failure
mode.

Two things in the same phase kept me honest about my own numbers. The confound
arm is one: I had written "reflection made it worse" before the control finished,
and the control showed half the damage was already in the reworded prompt with no
reflection in it.

The other is a false positive I nearly published. Re-running the control under a
new experiment name reproduced Phase 3.5's regret to sixteen digits, which I
recorded as confirming my design document's reproducibility claim. It did not:
those trajectories diverged at episode 1 and converged on the same answer because
language models propose round numbers. Checked properly, **1 of 10 runs
reproduced exactly and 3 of 10 matched on final score** — score agreement
overstated reproduction threefold. I then guessed the cause was the model not
fitting the GPU, measured that too, and was wrong again: an isolated request at a
fixed seed reproduces perfectly across a process restart on that same hardware.
So the claim that a pinned local model "reruns byte-identically" is retired, the
mechanism is narrowed to prompt-cache reuse across a long request sequence and
labelled a hypothesis, and what actually delivers reproducibility turns out to be
the recorded-call audit trail rather than the pinned model: replay is exact,
re-execution is not.
