# Phase 4 brief — the critic and the replanner

Written **before** building. The log (`phase-04.md`) gets written after, and it
must not be reconciled with this file afterwards: a brief quietly corrected in
hindsight stops being evidence of what was known beforehand. Phase 3.5's log
recorded two things this brief got wrong; the same rule applies here.

**This is the experiment the project exists to run.** `TECHNICAL_DESIGN.md` §13
calls Phase 4 *"the money experiment: does reflection pay?"*, and Phase 3.5
exists solely to make it answerable — on functions the model cannot have read
about, with a model that will still exist in two years.

---

## 1. The question, stated so it can be falsified

> Does an agent that **diagnoses its own results** and **re-plans from that
> diagnosis** find better designs, at an equal number of simulator calls, than
> the same agent with reflection removed?

`agent_no_reflection` is the control and already exists. `agent_full` is the
treatment. Everything else is held constant: same simulators, same seeds, same
evaluation budget, same guard, same model, same harness.

**A negative result is a result.** Phase 3.5's `agent_no_reflection` already
beats TPE at 20 evaluations on `branin_i1` (p=0.0159). Reflection has to move a
number that is already good, on a 20-evaluation budget where there is very
little room to be cleverer. It may well not, and the honest outcome is then
*"reflection did not pay at this budget on this model"* — which is a more useful
sentence than most positive results, because almost nobody measures it.

The one thing that must not happen is measuring reflection and reporting
something else. §5 below is about the two ways that could occur.

---

## 2. What gets built

| File | Responsibility |
|---|---|
| `agent/critic.py` | The critic. Explains a result it is *told* the grade of. |
| `agent/replanner.py` | The replanner. Chooses the next mode from the critic's output. |
| `prompts/critic.v1.md` | Critic template. |
| `prompts/replanner.v1.md` | Replanner template. |
| `prompts/planner.v2.md` | Planner + a reflection block. **v1 is not touched.** |

Changed: `strategies/base.py` (the `Reflects` optional protocol), `agent/loop.py`
(call the reflector, honour nothing it says), `strategies/llm_agent.py`
(`AgentFull`), `strategies/registry.py`, `agent/stagnation.py` (the two detectors
that needed a critic), `agent/state.py` (`CostRecord.plus`),
`eval/harness.py` (run locking — see §7), `eval/metrics.py` and `eval/report.py`
(reflection reporting), `cli.py`.

---

## 3. How reflection is wired, and why it is wired there

### The critic runs in the loop, not inside the strategy

`TECHNICAL_DESIGN.md` §3 puts `critic.evaluate` and `replanner.decide` in the
loop, after the simulator. That is also the only place they *can* go: the
`Strategy` protocol is `propose(goal, history, rng) -> Candidate` and returns
before the simulator has run. A critic living inside `propose` would be
critiquing the previous episode from inside the next one, which is the same
thing with the bookkeeping made confusing.

So the loop calls it — but the loop must not learn what a critic is, because the
same loop runs random search. The existing answer to that problem is
`ReportsCost`: an optional protocol the loop checks for at runtime, so baselines
do not implement an accounting method to report zero. Reflection gets the same
treatment.

```python
@runtime_checkable
class Reflects(Protocol):
    def reflect(self, goal, history, candidate, outcome, computed) -> Reflection: ...
```

`isinstance(strategy, Reflects)` decides whether an episode is reflected on.
`build_strategy(name)` stays the single source of truth for what a strategy is —
nothing in the harness or the CLI needs a second switch saying "and this one
also reflects", which is exactly the kind of split identity that later produces
a run labelled `agent_full` that quietly did not reflect.

### Rule 2 is enforced structurally, not by instruction

This is the most important design decision in the phase.

`Evaluation` has six fields. Three are computed from the simulator
(`improved`, `delta_vs_best`, `feasible`) and three are the model's
(`diagnosis`, `hypotheses`, `confidence`). The obvious implementation asks the
model for an `Evaluation` and trusts it not to touch the first three.

Instead, **the critic's schema does not contain them**:

```python
class CriticVerdict(BaseModel):      # what the LLM is asked for
    diagnosis:   str
    hypotheses:  list[str]
    confidence:  float
```

The loop then constructs the `Evaluation` itself, taking the computed three from
the oracle and the prose three from the verdict. The model is *shown* `improved`
and `delta_vs_best` in its prompt as given facts, and has no channel through
which to return a different value. Rule 2 stops being a convention the prompt
asks for and becomes a property of the type.

The same applies to the replanner, which returns `action`, `reason` and
`next_focus` — all three genuinely advisory, so all three may be the model's.

### Reflection reaches the next proposal through the database

`Episode.evaluation` and `Episode.decision` are already columns, already
persisted, and have been since Phase 1 — the schema was fixed then *specifically*
so that Phase 4 would need no migration. So the next planner prompt renders the
previous episode's diagnosis and decision straight out of `history`.

No new state, no field on the strategy object, no migration. Rule 1 survives
untouched: kill the process after episode 7 and resume, and episode 8's prompt is
rebuilt from the episodes table with the reflection in it.

### `TERMINATE` is recorded and not obeyed

`ReplanAction.TERMINATE` exists, and the replanner may return it. **The loop will
not act on it**, behind a `honour_terminate: bool = False` flag that stays off
for the whole experiment.

Two reasons, and the second is the real one:

1. Rule 2. Termination is a decision, and deterministic code makes decisions.
2. Fairness. A strategy that talks itself into quitting at evaluation 6 has not
   lost the same contest the others were in — it has set its own budget. The
   evaluation budget being identical across strategies is the single property the
   whole ablation rests on (`TECHNICAL_DESIGN.md` §6).

The rate at which the replanner *asks* to terminate is reported instead. That is
strictly more informative than letting it happen, because it is a number rather
than a confound.

The replanner's schema is narrowed to `EXPLOIT | EXPLORE | REPAIR | TERMINATE`.
`DECOMPOSE` and `ESCALATE` have no implementation behind them (§11.5, Phase 11),
and offering a model an action the system cannot perform is an invitation to
pick it.

### Failure of the critic must not lose a paid simulation

The critic and the replanner are two more model calls, and either can fail. If
one does, the episode is still written: the evaluation falls back to
`computed_only`, the decision to the placeholder, and the diagnosis records
`critic unavailable: <error>` so the failure is visible in the report rather than
looking indistinguishable from a run that had reflection switched off. The
simulator call has already been paid for and must not be thrown away because the
narration failed.

---

## 4. Prompt versioning, and the confound it creates

`agent_full`'s planner needs a reflection block in its prompt. Editing
`planner.v1.md` is not an option: it would change `call_key` for every recorded
Phase 3.5 call, making `phase35-main` unreplayable, and would silently alter
`agent_no_reflection` — the control — in the middle of the experiment measuring
it.

So `planner.v2.md` = v1 plus a `$reflection_block`, and `agent_no_reflection`
keeps v1.

**This creates a confound and it has to be stated plainly.** `agent_full`
differs from `agent_no_reflection` in *two* ways: it receives reflection content,
and it receives a differently-worded prompt. A win could be either.

The control for it is a third arm, `agent_prompt_control`: planner v2, empty
reflection block, no critic and no replanner calls. If `agent_prompt_control`
matches `agent_no_reflection`, the prompt change is inert and the `agent_full`
difference is reflection. If it does not, the headline has to be stated as
"reflection plus the prompt that carries it".

That arm is a seventh strategy, and `report.py::_slot_for` **raises** past six
series rather than reusing a colour — a guard Phase 3.5 added after two lines
came out the same blue. So the control runs as its own experiment,
`phase4-control`, reported separately. The guard is doing its job; working around
it by adding a seventh hue would be the wrong response to it.

---

## 5. The two ways this measurement could be wrong

Written now, before there is a number to protect.

**Reflection is not free, and the budget that is held equal hides that.** Every
`agent_full` episode makes three model calls where `agent_no_reflection` makes
one. At an equal *evaluation* budget that is invisible — which is correct,
because the premise of the project is that a simulator call costs minutes and a
token costs nothing. But it means "reflection pays" is a claim about sample
efficiency and not about efficiency in general, and the report must carry the
call and token counts next to the regret so nobody reads it the other way. This
is the same rescoping Phase 3.5 had to apply to the TPE comparison after
shipping; doing it in advance this time.

**Twenty evaluations may be too few for reflection to have anywhere to go.** The
sweep in Phase 3.5 found `agent_no_reflection`'s advantage over TPE significant
at 20 and gone by 40. Reflection plausibly needs *more* history to be worth
anything, not less, so 20 may be the budget at which it is least likely to show.
If the main matrix is null, the follow-up is a 40-evaluation cell rather than a
conclusion — and `sil-agent sweep` already exists to hold it. Phase 3.5's log
says this explicitly: the right question is *"at which budgets does reflection
pay?"*, not *"does it?"*.

---

## 6. The stagnation detectors that needed a critic

`agent/stagnation.py` implements two of the four detectors in
`TECHNICAL_DESIGN.md` §3 and says of the other two: *"they need a critic to
exist, so their interface is settled here and they arrive in Phase 4."* They
arrive here.

- **`ConfidenceDecline`** — critic confidence falling monotonically across a
  window. The agent saying, in a structured field, that it is running out of
  ideas.
- **`RepeatedDiagnosis`** — the last N diagnoses are near-identical. Measured on
  normalised token overlap rather than string equality, because a model
  restates the same thought in different words every time.

**They are implemented, unit-tested, and switched off for the experiment.** A
detector that fires at evaluation 12 gives `agent_full` twelve evaluations
against everyone else's twenty, which is the fairness problem from §3 arriving
by a different door. `run_loop` already takes `detectors=None` by default and
the Phase 2/3/3.5 matrices all ran that way; Phase 4 does the same. Their value
here is that they exist, are correct, and can be turned on for a single-run
demonstration outside the matrix.

---

## 7. Wiring `services/locks.py` into the harness

Phase 3 built run locking for exactly this case and Phase 3.5 hit the case
without it. From that log:

> Two processes were then walking the same experiment. [...] The irony is that
> Phase 3 built `services/locks.py` for exactly this and the harness does not use
> it — locking is wired for the CLI's single-run path, not for `execute_cell`.

What saved it was the `(run_id, idx)` natural key refusing the duplicate write,
which is safe but not graceful: the loser had already paid for a simulation and,
from Phase 3, a model call. At three model calls per `agent_full` episode that
waste triples.

The change is small:

- `execute_cell` wraps `run_loop` in `build_lock(cell.run_id)`.
- `LockUnavailable` marks the cell `"locked"` and moves on, rather than failing
  the matrix. A cell another process is working on should be skipped, not queued
  behind — `RunLock.acquire` already never blocks.
- `run_loop` gains an optional `heartbeat: Callable[[], bool]`, called after each
  episode is durably written. The harness passes `lock.renew`. A `False` means
  this worker no longer owns the run and must stop writing immediately, which
  terminates with `ERROR` — the same exit the duplicate-key path already uses.

`build_lock` returns a `NullLock` when `REDIS_URL` is unset or Redis is
unreachable, so nothing about running a single benchmark on a laptop changes.
That degradation is deliberate and predates this phase.

Note what locking does *not* do: it does not make the duplicate-key check
redundant. The lock has a TTL and a worker can stall past it; the natural key is
the thing that is actually load-bearing. The lock stops the waste, not the
corruption.

---

## 8. The experiment

New experiment **`phase4-main`**, deliberately the protocol `TECHNICAL_DESIGN.md`
§5 specifies and the one Phase 3.5 ran, so the numbers sit alongside rather than
replacing:

| | |
|---|---|
| Simulators | `branin_i1`, `hartmann6_i1` — shifted, rotated, unpublished |
| Seeds | **5** (1–5) — non-negotiable per §5 |
| Evaluations | 20 |
| Strategies | `random_search`, `grid_search`, `optuna_tpe`, `single_shot_llm`, `agent_no_reflection`, `agent_full` |
| Model | `qwen3:4b-q4_K_M`, local, pinned including quantisation |
| Test | Mann-Whitney U, two-sided, `method="exact"` |

Call budget:

| Strategy | Calls | Arithmetic |
|---|---|---|
| `agent_full` | 600 | 20 evals × 3 calls × 5 seeds × 2 sims |
| `agent_no_reflection` | 200 | 20 × 1 × 5 × 2 |
| `single_shot_llm` | 10 | 1 × 5 × 2 |
| baselines | 0 | |
| **total** | **≈810** | ~4–6 hours at a 2.2 s warm call, longer prompts allowing |

Then **`phase4-control`**: `agent_prompt_control` and `agent_no_reflection`, same
two simulators, same five seeds, same budget. 200 further calls.

### A free reproducibility check

`agent_no_reflection` re-runs under a new experiment name, so it gets new
`run_id`s, so the replay cache misses and every call is made again — same prompt
(v1, untouched), same seed reaching the sampler, same pinned model. Phase 3.5's
central claim is that this reproduces **byte-identically**.

So the phase gets a test of that claim for free: `phase4-main`'s
`agent_no_reflection` regrets should equal `phase35-main`'s exactly. If they do,
the reproducibility argument for going local is demonstrated rather than
asserted. If they do not, that is a more important finding than the phase's
headline and gets reported as such.

---

## 9. What I expect to go wrong

Written in advance so the log can say which of these actually happened.

1. **The critic writes fluent, useless diagnoses.** A 4B model told an objective
   went from 6.1 to 5.8 will say "the reduction suggests the region is
   promising" for every result it ever sees. The failure mode is not an error —
   it is prose that reads fine and steers nothing, and `RepeatedDiagnosis` is
   the instrument for detecting it.
2. **Reflection makes it worse.** A confident wrong diagnosis biases the next
   proposal, where `agent_no_reflection` would simply have looked at the
   numbers. Entirely plausible at 4B, and a legitimate result.
3. **Three calls per episode is slower than estimated.** The critic prompt
   carries the result and the history; the replanner's carries the critic's
   output. Both are longer than the planner's, and `num_ctx` is 4096.
4. **Prompt-fits-context refusals on `hartmann6_i1`** — six parameters, twenty
   history lines, plus a diagnosis block. The Ollama adapter refuses rather than
   letting Ollama truncate silently, so this appears as failed episodes, not as
   quiet nonsense. Mitigation is `BEST_SHOWN`/`RECENT_SHOWN` on the critic path,
   not a larger context.
5. **The prompt control is not inert**, and the headline has to be widened to
   "reflection and the prompt that carries it".
6. **Null result at 20 evaluations.** See §5. The response is a sweep cell at 40,
   not a re-interpretation of the null.

---

## 10. Deliverables

- [ ] `docs/phases/phase-04-brief.md` (this file)
- [ ] `agent/critic.py` — `CriticVerdict` without the computed fields
- [ ] `agent/replanner.py` — narrowed action set
- [ ] `prompts/critic.v1.md`, `prompts/replanner.v1.md`, `prompts/planner.v2.md`
- [ ] `Reflects` protocol; `run_loop` builds the `Evaluation` from oracle + verdict
- [ ] `AgentFull`, registered; `agent_prompt_control`, registered
- [ ] `TERMINATE` recorded, not obeyed; recommendation rate reported
- [ ] `ConfidenceDecline` and `RepeatedDiagnosis`, tested, off in the matrix
- [ ] `services/locks.py` wired into `execute_cell`, with heartbeat renewal
- [ ] Reflection and cost-of-reflection tables in the report
- [ ] Prompt-anonymity test extended to the critic and replanner prompts
- [ ] `phase4-main` executed; `phase4-control` executed
- [ ] Reproducibility check against `phase35-main` reported
- [ ] `docs/phases/phase-04.md`, six sections, "what went wrong" non-empty
- [ ] One commit for the phase, per the standing convention

## Out of scope

Episodic memory across runs (Phase 5). Budget governor, OTel, MCP, FastAPI
(Phase 6). `DECOMPOSE` and `ESCALATE` (Phase 11). Any paid model — the frontier
comparison §5 calls for is worth doing once the local pipeline has produced a
result worth comparing against, and not before.
