# Phase 2 brief — baselines, eval harness, ablation report

Written **before** building. The log (`phase-02.md`) gets written after.

**No LLM code in this phase.** Still none. Phase 2 is the last phase where every
number is exactly reproducible, and that is the whole point of it: the measuring
instrument gets built and calibrated before the thing being measured exists.

---

## Goal of this phase

Produce **a comparison table nobody can poke a hole in.**

Phase 1 proved a run survives being killed. Phase 2 proves a *result* survives
being questioned. The specific claim the project will eventually make is "the
agent beats classical search", and an interviewer's first three questions will be:
how many seeds, was the budget identical, and how do you know that gap isn't
noise. This phase makes all three answerable before there is anything to defend.

Concretely: three non-LLM strategies × three benchmarks × N seeds, run through
the identical harness, scored on **regret against the known optimum**, reported
with dispersion, from SQL over the `episodes` table.

The three LLM rows (`SingleShotLLM`, `AgentNoReflection`, `AgentFull`) are
Phases 3–4. Build the harness so they drop in as three more `Strategy`
implementations and *nothing else changes*. If adding an LLM row later requires
touching `eval/harness.py`, the harness was built wrong.

---

## What to build

### 1. `GridSearch` (`strategies/grid.py`)

Naive systematic coverage. The interesting part is not the grid, it is making a
stateful-feeling algorithm fit a stateless protocol.

`Strategy.propose(goal, history, rng)` gets no counter. The grid position must
come from `history` — `recompute_step_idx(history)` is already in
`agent/loop.py` and does exactly this. **Never** hold an index on the strategy
object; that is the hidden in-memory state Rule 1 forbids, and it would break
resume silently (see `strategies/base.py` for the same argument applied to RNGs).

**Size the grid to fit the budget.** With `d` dimensions and budget `N`, use
`points_per_dim = floor(N ** (1/d))`, giving `points_per_dim ** d ≤ N`. For
Branin at N=200 that is 14×14 = 196. For Hartmann-6 at N=200 it is 2⁶ = 64 —
grid search gets through a third of its budget and stops. That is not a bug to
fix, it is the curse of dimensionality showing up on your own table, and it
should be reported as such.

The alternative — enumerate a grid larger than the budget and truncate — is
worse and worth understanding why. Odometer enumeration varies the last
parameter fastest, so a truncated grid leaves the first parameter pinned near its
lower bound for the entire run. The result looks like a grid search but is
actually a badly biased line search, and the table would be quietly wrong.

Grid points at the bounds themselves (`linspace(low, high, k)`, endpoints
included), not cell centres. Both are defensible; endpoints are what a person
does by hand. Document the choice, because it matters: Rosenbrock's optimum is at
xᵢ = 1 and a 3-point grid over [-5, 10] visits -5, 2.5, 10 and never gets close.

**The grid runs out before the budget does.** The loop currently has no way for a
strategy to say "I have nothing left to propose" — `TerminationReason` covers
SUCCESS, BUDGET, STAGNATION, ERROR. Add a `StrategyExhausted` exception raised by
`propose`, caught in `run_loop`, terminating with a new `EXHAUSTED` reason. Small
change to Phase 1 code; make it deliberately and note it in the log.

### 2. `OptunaTPE` (`strategies/optuna_tpe.py`)

The honest competitor, and the hardest thing in this phase.

Optuna's `Study` object is stateful: it accumulates trials in memory and the
sampler reads them. Holding one on the strategy is the obvious implementation and
it will pass every test you write, right up until a run is resumed in a fresh
process — at which point episode 40 is proposed by a sampler that has seen zero
trials, and the run silently becomes a different run. This is the same failure
mode as the per-run RNG in Phase 1, in a new costume.

**So rebuild the study from `history` on every call:**

```
propose(goal, history, rng):
    study = optuna.create_study(direction=..., sampler=TPESampler(seed=derived))
    for episode in history:
        if episode.sim_result is not None:
            study.add_trial(create_trial(params=..., distributions=..., value=...))
    trial = study.ask()
    return Candidate(params=trial.params, source=BASELINE)
```

Four details that will each cost an hour if missed:

- **Distributions come from `ParameterSpace`.** Map `ParamKind.FLOAT` →
  `FloatDistribution(low, high)`, `INT` → `IntDistribution`, `CATEGORICAL` →
  `CategoricalDistribution`. The simulator stays the authority on what exists —
  same rule as the guard.
- **Seed the sampler from persisted values**, e.g. the same `f"{seed}:{idx}"`
  derivation `episode_rng` uses, hashed to an int. Optuna's sampler has its own
  RNG; leaving it unseeded reintroduces exactly the non-determinism this design
  works to avoid.
- **Rebuilding is O(n) per episode**, so a 200-episode run does ~20,000 trial
  reconstructions. At this scale that is fine and correctness wins. Measure it and
  put the number in the log — it is the honest cost of Rule 1 and worth quoting.
- **Failed episodes are skipped**, not added as failed trials. `sim_result` is
  `None` for guard rejections; baselines produce none, but the code path exists
  from Phase 3.

**Constraints.** `branin_constrained` needs `TPESampler(constraints_func=...)`,
where the function returns violation amounts (≤ 0 meaning satisfied). Since trials
are reconstructed anyway, store the amounts from `SimResult.constraint_violations`
on each rebuilt trial and read them back. If this fights you for more than an
afternoon, run TPE on the unconstrained benchmarks only and record the omission
in the table as a gap rather than papering over it. A missing cell is honest; a
cell computed differently from its neighbours is not.

### 3. The evaluation harness (`eval/harness.py`)

Executes the matrix: strategies × benchmarks × seeds. It calls `run_loop` and
otherwise stays out of the way.

**Make run identity deterministic.** Give the harness an experiment name and
derive each `run_id` with UUIDv5 from `(experiment, strategy, simulator, seed,
max_evaluations)`. UUIDv5 hashes a name into a UUID, so the same tuple always
produces the same id.

This is the whole design of the harness in one decision. It means an interrupted
harness invocation is restarted by *re-running the identical command*: each run
either does not exist yet (create it), or exists and is incomplete (load,
`rehydrate`, continue), or is finished (skip). No job table, no progress file, no
resume flag — Phase 1's idempotent `append_episode` and recompute-on-load already
do all of it. Ninety runs can be started before bed and the laptop can sleep.

It also makes a subtle mistake impossible: you cannot accidentally run the same
configuration twice and average four seeds worth of data into a five-seed cell,
because the second attempt lands on the same `run_id`.

Requires a small addition to `RunRepository`: a way to list runs for an
experiment. Add an `experiment` column (nullable text, indexed) to `runs` **via
an Alembic migration** — this is the first schema change against a table that
already holds rows, which is exactly what the migration machinery was set up for
in Phase 1.

### 4. Scoring (`eval/metrics.py`)

Everything here is deterministic and unit-testable. Keep it separate from both
the harness and the report so it can be tested without a database.

**Primary metric: regret.** `best_feasible_objective - known_optimum` for a
MINIMISE benchmark. `known_optimum` is already on `Benchmark` in `simulators/toy.py`.

Three rules to settle now:

- **Only feasible results count** towards best-found. A run with no feasible
  result has undefined regret — report it as such, never as a large number.
  Silently treating infeasible-but-good as a result would flatter whichever
  strategy ignores constraints.
- **Rosenbrock needs a log scale.** Regret ranges over ~5 orders of magnitude
  there (Phase 1's random search scored 235.6). Report `log10(regret)` alongside
  raw, and use the log scale in plots or the curves are unreadable.
- **Aggregating across benchmarks needs normalisation**, since Branin regret ~0.1
  and Rosenbrock regret ~235 cannot be averaged. Normalise per benchmark against
  the random-search median at full budget, or don't aggregate at all. Prefer not
  aggregating: three separate rows say more than one meaningless mean.

**Secondary metrics**, each cheap and each answering a question that will be
asked:

| Metric | Question it answers |
|---|---|
| Evaluations to reach a regret threshold | Budget efficiency — the number Phase 7's surrogate must improve |
| Best-so-far curve per evaluation | Does it converge, or get lucky once? |
| Fraction of feasible evaluations | Does it respect constraints or stumble into them? |
| Guard rejections (0 for all baselines) | Reserved: from Phase 3, how often the LLM proposes nonsense |
| Wall-clock per episode | Establishes the pre-LLM floor, so Phase 3's latency is attributable |

### 5. The ablation report (`eval/report.py`)

Reads from Postgres through the repository, writes Markdown plus plots to
`reports/<experiment>/`. No log scraping — the episodes table is the record.

Output:

- **Headline table**: strategy × benchmark, cells showing `mean ± std` regret
  over seeds, plus median and n.
- **Per-seed appendix**: every raw number. If the headline table is ever
  disputed, this is the answer. Cheap to produce and it removes any suspicion of
  cherry-picking.
- **Convergence plots**: best-so-far regret vs evaluation count, one panel per
  benchmark, median line with an interquartile band across seeds. Not mean ±
  std here — one unlucky seed on Rosenbrock distorts a mean badly.

Adds `optuna`, `matplotlib` and `scipy` as dependencies.

### 6. Statistics — and how far not to push them

`TECHNICAL_DESIGN.md` §9 locks the protocol at ≥5 seeds, mean ± std. Keep that as
the cross-strategy protocol: it is what the LLM strategies can afford in Phases
3–4 on free tiers, and the comparison must be like for like.

But baselines cost €0.00 and run in seconds, so **also run each baseline at 30
seeds** and use those runs for one purpose only: measuring the **seed noise
floor**. Bootstrap 5-seed samples out of the 30 and report the spread of the
5-seed mean. That produces a sentence worth more than any p-value:

> "With 5 seeds, the 90% interval on mean Branin regret is ±X. A gap smaller than
> X is not evidence of anything."

Write that number down in Phase 2, before there is any agent result to be excited
about. It is the guardrail for Phase 4, and it is much easier to set honestly now.

For strategy-vs-strategy comparison use **Mann-Whitney U** (a rank test — it asks
whether one strategy's results tend to sit above the other's, without assuming
the distribution is normal, which regret distributions are not). With n=5 per
group the smallest achievable two-sided p is 0.0079, so the test is usable but
detects only large effects. Say so. Do not report a t-test on five skewed
samples.

### 7. Tests

- Grid search covers the full grid, in order, and never repeats a point
- Grid position derives from history — construct a fake 7-episode history, assert
  the 8th proposal is the 8th grid point
- `StrategyExhausted` terminates the loop cleanly with the run recorded
- TPE proposes an identical sequence given the same seed
- **TPE resume equivalence**: run 40 episodes uninterrupted; run the same
  configuration with `--crash-at 20` and resume; assert the two episode sequences
  are byte-identical. This is the test the whole phase rests on, and the one that
  catches the stateful-study mistake.
- Regret is `None`, not a large float, when no feasible result exists
- Harness re-invocation is idempotent — run the matrix twice, get the same row
  count and the same numbers
- Metrics functions against hand-computed values

---

## Fairness rules — lock these now, in writing

These are the answers to the interview questions. Fix them while there is no
result to be tempted by.

1. **Equal evaluation budget.** Every strategy gets the same
   `max_evaluations` on the same benchmark. This is why the budget counts
   simulator calls and not episodes.
2. **Same seed set.** Seeds 1–5 for the cross-strategy table, everywhere. Not
   "five seeds I picked".
3. **Same benchmarks, same parameter spaces**, straight from `describe()`.
4. **Failed episodes are reported, never dropped.** A strategy that produced 30
   rejections and 200 evaluations does not get to show only the 200.
5. **The report is generated, never hand-edited.** If a number looks wrong, fix
   the code and regenerate.

**One inconsistency to resolve while it is still cheap.** `run_loop` currently
terminates on `state.step_idx >= budget.max_evaluations`, and `step_idx` counts
*episodes*, including guard rejections — while `budget.evaluations_used` counts
only real simulator calls. The two disagree the moment a proposal is rejected.
Nothing in Phases 1–2 can produce a rejection, so this is invisible today and
will not be from Phase 3, when it would quietly hand a hallucinating planner
fewer simulator calls than a well-behaved one and bias the headline result.

Fix it here: terminate on `evaluations_used >= max_evaluations`, and add a
separate episode cap (`max_evaluations + max_rejections`) so a strategy that
proposes nothing valid cannot loop forever.

---

## Acceptance criteria

Phase 2 is done when all of these hold:

1. `eval/harness.py` runs 3 strategies × 3 benchmarks × 5 seeds = 45 runs
   unattended, from one command
2. Killing the harness mid-matrix and re-running the identical command completes
   it — no duplicated runs, no lost episodes, no extra flags
3. `eval/report.py` emits the ablation table, the per-seed appendix and the
   convergence plots, entirely from database reads
4. TPE resumed from a crash produces a sequence identical to an uninterrupted run
5. Grid search on Hartmann-6 terminates as EXHAUSTED at 64 evaluations and the
   report says so rather than showing a blank
6. The seed noise floor is quantified and written down
7. Phase 1's random-search numbers are reproduced under the locked protocol —
   they were single runs at mixed budgets (Branin 200, Hartmann-6 400,
   Rosenbrock 500) and are not directly comparable to anything yet
8. `ruff` clean, `mypy` clean, `pytest` green

Criterion 4 is the one that matters. Criterion 2 is the one that will save a
weekend.

---

## Known traps

Phase 1's Hartmann-6 warning paid for itself, so:

- **The stateful Optuna study.** Covered above. It will pass your tests.
- **Optuna logs at INFO on every trial.** 45 runs × 200 episodes of "Trial 0
  finished with value…" will bury everything. `optuna.logging.set_verbosity`.
- **`study.ask()` mutates the study.** Rebuilt fresh each call here, so it does
  not matter — but do not optimise the rebuild away without re-reading this.
- **Matplotlib picks an interactive backend** and will hang or warn under
  pytest/CI. Force `Agg`.
- **Regret can go slightly negative** if `known_optimum` is a rounded literal
  (Branin's 0.397887, Hartmann's -3.32237). Clamp at zero for display, and do not
  let it look like a strategy beat the global optimum.
- **`ruff format .` reformats Markdown.** Already prevented in `pyproject.toml`;
  do not undo it while adding config for new tools.

---

## What this unlocks

Phase 3 gets to add an LLM strategy and immediately know whether it is any good,
against a table that already exists and was written without knowing what the
answer would be. Phase 4 — the experiment this project exists to run — gets a
noise floor, so "reflection helped" can be distinguished from "that seed was
lucky".

Phase 7's surrogate claim ("cut simulator calls by 60% at equal quality") is
measurable only because *evaluations-to-threshold* is defined and baselined here.

---

## Explicitly out of scope

- Any LLM call, any LLM SDK, any prompt
- Budget governor beyond the evaluation count (Phase 6)
- Tracing, OpenTelemetry, Langfuse (Phase 6)
- FastAPI, MCP (Phase 6)
- Stagnation detection (Phase 3)
- Redis locking (Phase 3)
- Surrogate models (Phase 7), multi-objective (Phase 8)

The temptation this phase is to skip ahead to the agent, because the agent is the
interesting part. The value of Phase 2 is precisely that it is finished before
the interesting part starts and therefore cannot be tuned to flatter it.

---

## Finally

Write `docs/phases/phase-02.md` using the template in `README.md`. All six
sections, including "what went wrong" — and record the actual numbers, including
any case where a baseline beat what you expected the agent to have to beat.
