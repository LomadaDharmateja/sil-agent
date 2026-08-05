# Phase 2 — baselines, eval harness, ablation report

Built 2026-08-05. Still no LLM code, by design.

## 1. Why this phase exists

The project's claim is that an LLM agent optimises an engineering design better than
classical search. An interviewer's first three questions about that claim will be: how
many seeds, was the budget identical, and how do you know the gap isn't noise. This
phase makes all three answerable *before* there is anything to defend.

Phase 1 proved a run survives being killed. This one proves a result survives being
questioned. The order matters: the measuring instrument is built and calibrated while
nobody knows what it is going to say, so it cannot be tuned — consciously or otherwise —
to flatter the thing it will eventually measure. Whatever Phase 4 reports about
reflection, the bar it is measured against was set today.

Three of the six strategies in `TECHNICAL_DESIGN.md` §9 are non-LLM and exist now. The
other three arrive in Phases 3–4 as three more implementations of the same protocol. If
adding one had required touching the harness, the harness would have been built wrong.

## 2. What I built

A command that runs every strategy against every benchmark at every seed, survives being
interrupted, and produces a report with the raw numbers attached.

| File | Responsibility |
|---|---|
| `sil_agent/strategies/grid.py` | Grid search, sized to fit the evaluation budget. |
| `sil_agent/strategies/optuna_tpe.py` | Optuna's TPE sampler, rebuilt from history on every call. |
| `sil_agent/strategies/registry.py` | Name → strategy, so a stored run can reconstruct its own strategy. |
| `sil_agent/eval/metrics.py` | Regret, curves, aggregation, the seed noise floor. No database, no files. |
| `sil_agent/eval/harness.py` | The matrix runner. Deterministic run identity. |
| `sil_agent/eval/report.py` | Tables and convergence plots, generated from SQL. |
| `sil_agent/persistence/migrations/versions/78e3…` | Adds `runs.experiment`. |
| `tests/test_strategies.py` | Grid coverage and sizing; TPE determinism and resume equivalence. |
| `tests/test_metrics.py` | Scoring, against hand-computed values. |
| `tests/test_harness.py` | Run identity; restart by re-invocation, with a real crash. |
| `tests/test_loop_termination.py` | The budget-accounting fix and the new exits. |

Changed from Phase 1: `agent/loop.py` (terminates on evaluations, not episodes; catches
`StrategyExhausted`), `agent/state.py` (rejection counters, two new termination reasons),
`persistence/repo.py` (`experiment` column, `list_runs`), `cli.py` (`ablate`, `report`).

## 3. How it works

### The one idea in the harness

Each cell of the matrix gets a `run_id` derived with UUIDv5 from
`(experiment, strategy, simulator, seed, max_evaluations)`. UUIDv5 hashes a name into a
UUID, so the identifier is a *function of the configuration* rather than something freshly
invented each time.

Everything else falls out of that. A cell is either absent (create it), present but
unfinished (load, rehydrate, continue) or finished (skip). So an interrupted 60-run matrix
is restarted by issuing the identical command again — no job table, no progress file, no
`--resume` flag, because there is nothing to keep track of. Phase 1's idempotent
`append_episode` and recompute-on-load do the rest.

It also makes a quiet mistake impossible. Running the matrix twice cannot produce two sets
of runs for one configuration and average nine seeds' data into a five-seed cell, because
the second attempt lands on identifiers that already exist.

The price: changing the budget changes every identifier, so a 200-evaluation experiment and
a 400-evaluation one never collide. That is intended — they are different experiments and
pooling them would be wrong.

### Making TPE stateless

Optuna is built around a `Study` object that accumulates trials, and this codebase forbids
exactly that. The obvious implementation holds one on the strategy; it passes every test
that does not restart the process, and then fails silently. Resume a run in a fresh process
and the sampler starts with zero trials, so episode 40 is proposed as though it were
episode 0. The run completes, the numbers look plausible, and it is no longer the run it
would have been.

So the study is rebuilt from `history` on every call: every completed episode is replayed
into a fresh study as a finished trial, the sampler is seeded from persisted values, and
one point is asked for.

```python
sampler = TPESampler(seed=rng.getrandbits(32), constraints_func=...)
study = optuna.create_study(direction=..., sampler=sampler)
study.add_trials([create_trial(params=…, value=…, distributions=…) for e in history])
trial = study.ask(distributions)
```

The `rng` is already `Random(f"{seed}:{idx}")` from Phase 1, so drawing the sampler's seed
from it inherits that determinism. Cost: one proposal takes 0.7 ms with one trial in
history and 23.6 ms with two hundred — 1.85 s over a full run. That is what Rule 1 costs
here, and it is worth it.

### Grid search, and why it is allowed to lose badly

With `d` parameters and budget `N`, each axis gets `k = floor(N ** (1/d))` points, so the
grid is `k**d ≤ N` and always completes. Branin gets 14×14 = 196 of its 200. Hartmann-6
gets 2⁶ = 64 — two points per axis is all a 200-evaluation budget buys in six dimensions,
and two points on `[0, 1]` are 0 and 1, so grid search evaluates **only the 64 corners of
the unit hypercube**. Its best corner scores −0.166 against a true optimum of −3.322.

I verified that independently rather than trusting it:

```
best of all 64 corners (computed directly): -0.165567
best of the grid the strategy enumerates:   -0.165567
```

The rejected alternative looks more thorough and is much worse: build a grid larger than
the budget and let the run truncate it. Odometer enumeration varies the last parameter
fastest, so a grid truncated at 200 of 4096 points leaves the first parameter pinned at its
lower bound for the whole run. It would be labelled "grid search" and would actually be a
line search along one axis.

Because grid search stops early, the loop needed a way for a strategy to say "I have
nothing left" — a `StrategyExhausted` exception and a new `EXHAUSTED` termination reason,
reported separately from `BUDGET`. "Used 64 of 200 because it finished" and "used all 200"
are different facts and the table has to say which.

### The seed noise floor

The protocol is five seeds, per `TECHNICAL_DESIGN.md` §9, because that is what the LLM
strategies will be able to afford on free tiers. But baselines cost nothing, so each was
also run at 20 seeds — used for one purpose only: repeatedly drawing 5 of those 20 and
measuring how much the 5-seed mean moves when *nothing differs but the seeds*.

On Hartmann-6, TPE's five-seed mean regret ranges over [0.086, 0.241] — a spread of 0.155
against a mean of 0.156. **A Phase 4 agent that beats TPE on Hartmann-6 by less than about
0.15 has demonstrated nothing.** Writing that down now, before there is a result to be
excited about, is the whole point.

## 4. Key decisions and trade-offs

**Regret over feasible results only, and `None` when there are none.** A run with no
feasible result scores `None`, not a large float. A large float can be averaged and the
average would be meaningless; `None` forces the report to say "no feasible solution found".
Letting an infeasible-but-low value count would flatter precisely the strategy that ignores
constraints, which is what the constrained benchmark exists to catch.

**The termination reason is derived, not stored.** It is fully determined by status,
budget and history, and a derived value that is *also* persisted is a value that can
disagree with the record — the same argument that made `best` and `step_idx` recomputed in
Phase 1. It also avoided a second migration. The one case that needs care: an unfinished
run must report `None` rather than `EXHAUSTED`, or the harness would skip it instead of
resuming it. There is a test for exactly that.

**Sequential execution.** The simulators are microseconds and the bottleneck is two
database round trips per episode. Parallelism would save minutes at the cost of concurrent
writers, which the loop does not yet guard against (Redis locking is Phase 3). An ablation
that takes twenty minutes and is obviously correct beats one that takes five and needs an
argument.

**Four benchmarks instead of three.** The brief said three. `branin_constrained` was added
because it is the only one that exercises the feasibility path — violations, infeasible
ordering, constraint-aware sampling — for the same cost. It earned its place immediately.
Random search draws the identical points on Branin and on constrained Branin, since the
space and the seed are the same; the scores diverge only at seed 5, where the best
unconstrained point violates `x1 + x2 ≤ 10` and the constrained run correctly falls back to
a worse feasible one (0.529 → 0.806). That single differing cell is the feasibility path
proving it works end to end.

**What would have been easier but worse:** letting guard rejections come out of the
evaluation budget. It is one counter instead of two, and it is invisible in Phase 2 because
nothing here can propose an invalid candidate. It would also have meant that from Phase 3, a
planner that proposed thirty pieces of nonsense would get thirty fewer *simulator calls*
than a well-behaved one, and would then lose the comparison partly because of the
accounting rather than the search.

**What would be better but too expensive right now:** a proper effect-size estimate with
confidence intervals per comparison, rather than a rank test plus a separately-measured
noise floor. The noise floor answers the same question well enough at five seeds, and the
honest limitation — that five seeds detects only large effects — is stated rather than
engineered around.

### Deviations from the brief

1. **`build_strategy` takes the evaluation budget.** Grid search must know its budget to
   size its grid, and `Strategy.propose` is not given one. It comes from
   `BudgetState.max_evaluations`, which is persisted — so a resumed run rebuilds an
   identical grid. Had it come from a command-line flag, resuming with a different
   `--episodes` would silently switch to a different grid.
2. **Noise floor at 20 seeds, not 30, and only for the stochastic strategies.** 20 seeds
   gives 15,504 distinct 5-subsets, which is ample. Grid search is deterministic, so its
   noise floor is exactly zero by construction and running it 20 times would have measured
   nothing at a cost of 20 minutes.

## 5. What went wrong

**The rank test was quietly lying about grid search.** The first report showed
`grid_search vs random_search` on Branin at p = 0.0075 — the smallest p the test can
produce at n=5. Grid search is deterministic: its five seeds are five *identical* runs.
Those are not five independent observations, they are one observation written down five
times, and Mann-Whitney cannot tell replication from repetition. The p-value was real
arithmetic on fake replication. Fixed by detecting zero-variance cells and flagging those
rows with a footnote telling the reader to compare medians and disregard the p-value. The
uncomfortable part is that the number looked *better* than the honest ones and I nearly
kept it — a table of small p-values is exactly what you want to see.

**A test asserted something it never tested.** `test_an_interrupted_cell_is_continued_not_restarted`
passed on the first run. It contained the comment "execute, then truncate the history to
simulate dying partway through" and no truncation — it ran a cell to completion, ran
another to completion, and compared them. Green, meaningless. Rewritten to use the
`CrashingRepository` device from Phase 1: kill the cell at episode 5, verify five episodes
are durably committed, verify the run does *not* look finished, re-invoke, and assert it
continued at 5 and matched an uninterrupted run byte for byte. It is now parameterised over
TPE as well, because TPE is the strategy whose natural implementation breaks precisely
there. Lesson: a test that passes immediately on code you have not yet exercised deserves
to be read again.

**The chart labels overlapped and I only saw it by looking.** Two strategies converging to
similar regret put their direct labels on top of each other, and `tight_layout` clipped
them off the image edge. Nothing failed; the file was written and the code was clean. Fixed
by laying labels out in axis-fraction space with a minimum separation and reserving the
right fifth of the figure. The failure mode is worth remembering: the plotting code was
correct, the palette was validated, and the output was still unreadable — no check catches
that except rendering the image and looking at it.

**Grid search's floating-point sizing silently wasted evaluations.** `196 ** 0.5` evaluates
to `13.999999999999998`, which truncates to 13 — a 13×13 grid, 169 points, 27 evaluations
of the budget unused for no reason. Caught because the sizing test asserted 14 and got 13.
Fixed by stepping up while the larger grid still fits. It would never have raised an error;
grid search would just have been slightly, invisibly worse.

**Pydantic warned me off a clever helper.** I first wrote `BudgetState._replace(**changes)`
to avoid repeating eight constructor arguments in two methods. It needed a
`# type: ignore[arg-type]` because the kwargs dict types as `float | int` while
`max_evaluations` is `int`. A type-ignore to save six lines of explicit code is a bad
trade in a codebase whose stated preference is clarity over cleverness, so I wrote both
constructors out in full.

**The test database was one migration behind.** `alembic upgrade head` uses `DATABASE_URL`,
which points at the development database. The test database is a *different* database, and
the db-marked tests failed against it with "column runs.experiment does not exist" until it
was migrated separately via `ALEMBIC_DATABASE_URL`. Phase 1's `env.py` had already
anticipated this and documented the variable; I just had not needed it before. Worth
recording because it will recur on every schema change and in CI.

**Optuna's experimental warning survived a filter.** `constraints_func` is marked
experimental and warns on every sampler construction — once per episode, so 200 identical
warnings per constrained run. A module-level `warnings.filterwarnings` silenced it at
runtime but *not* under pytest, which installs its own filters around each test and
overrides it. The filter had to be repeated in `pyproject.toml`. Both are narrow: this one
message, not experimental warnings in general.

## 6. What this unlocks

Phase 3 can add an LLM strategy and immediately know whether it is any good, against a
table that already exists and was produced without knowing what the answer would be. It
inherits, specifically:

- **`StrategyExhausted` and the corrected budget accounting**, so a planner that
  hallucinates is measured on the same number of simulator calls as one that does not.
- **The harness**, which needs one new entry in `STRATEGY_FACTORIES` and nothing else.
- **The noise floor**, so Phase 4 can distinguish "reflection helped" from "that seed was
  lucky" — the single most important input to the experiment this project exists to run.
- **Evaluations-to-threshold**, baselined here, which is the metric Phase 7's surrogate
  claim ("60% fewer simulator calls at equal quality") has to move.

One thing to fix before Phase 4: the report's palette is CVD-safe for three series and
Phase 4 has six. That needs small multiples, not three more hues.

## Numbers

| Measurement | Value |
|---|---|
| Runs executed this phase | 220 (60 main + 160 noise) |
| Episodes written this phase | 42,835 (10,685 main + 32,000 noise + 150 smoke) |
| Episodes in the database | 43,991, including Phase 1's CLI runs |
| Proposal cost, random search | 0.03 ms |
| Proposal cost, grid search | 0.05 ms |
| Proposal cost, TPE | 9.3 ms mean; 0.7 ms at 1 trial → 23.6 ms at 200 |
| TPE study-rebuild overhead per run | 1.85 s |
| Tests | 137 (83 from Phase 1, 54 new), green, 18.5 s |
| Code | 37 Python files, ~5,230 lines; ~2,170 added this phase |
| LLM spend | €0.00 |

Final regret over 5 seeds at 200 evaluations, lower is better:

| Strategy | Branin | Branin (constrained) | Hartmann-6 | Rosenbrock (4-D) |
|---|---|---|---|---|
| Grid search | 0.0204 | 0.0204 | 3.157 | 4226 |
| Random search | 0.153 ± 0.12 | 0.208 ± 0.16 | 0.866 ± 0.42 | 547 ± 830 |
| **Optuna TPE** | **0.0188 ± 0.024** | **0.0092 ± 0.013** | **0.152 ± 0.10** | **28.9 ± 33** |

Evaluations actually used: grid search 196, 196, **64**, **81**; everything else 200.

Seed noise floor — the 90% interval of a 5-seed mean, from 20 seeds:

| Strategy | Benchmark | Mean | 90% interval | Spread |
|---|---|---|---|---|
| optuna_tpe | branin | 0.0084 | [0.0029, 0.0185] | 0.0156 |
| optuna_tpe | hartmann6 | 0.156 | [0.086, 0.241] | 0.155 |
| optuna_tpe | rosenbrock | 25.4 | [9.2, 47.8] | 38.6 |
| random_search | hartmann6 | 0.920 | [0.721, 1.133] | 0.412 |
| random_search | rosenbrock | 769 | [283, 1280] | 997 |

Three things stand out.

**Grid search wins in 2-D and collapses in 6-D.** It is competitive with TPE on Branin
(0.0204 vs 0.0188, and the difference is inside TPE's own noise floor) and 20× worse than
random search on Hartmann-6. Same algorithm, same budget, same code.

**Rosenbrock remains the discriminator.** TPE reaches regret 29 where random search reaches
547, but the true optimum is 0 and neither is close. Random search finds the valley and has
no way to follow it; TPE follows it slowly. This is the benchmark where an agent that
reasons about *why* a result came out as it did has room to show something, and where the
answer will not be ambiguous.

**TPE's advantage is real but not overwhelming.** On Branin it beats random search by 8×
and grid search not at all. The noise floor explains why that has to be checked rather
than eyeballed: on Rosenbrock TPE's five-seed mean can land anywhere in [9, 48].

## Interview angle

I can show a comparison table that answers the three questions before they are asked: five
seeds with every raw number published, an identical evaluation budget enforced by the loop
rather than by convention, and a measured noise floor saying how large a difference has to
be before it means anything. The harness that produced it restarts from an interruption by
re-running the same command, because run identity is a UUIDv5 of the configuration rather
than something freshly generated.

The sharper story is the one that did not make it into the table. My first report had grid
search beating random search at p = 0.0075 — the most significant result the test could
produce. Grid search is deterministic, so its five seeds are one run recorded five times;
the test was measuring repetition and calling it replication. The report now detects
zero-variance cells and tells the reader to disregard those p-values. It is a better
project for having caught that before Phase 4, when the same mistake would have been made
about the agent, in the direction I wanted to believe.
