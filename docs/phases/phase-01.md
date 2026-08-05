# Phase 1 — state, persistence, toy simulator

Built 2026-08-04. No LLM code, by design.

## 1. Why this phase exists

The project's claim is that an LLM agent can optimise an engineering design better
than classical search. Any such claim is only as good as the measurement behind it,
and the measurement needs two things that do not exist by default: runs that survive
being interrupted, and runs that repeat exactly when given the same seed.

Phase 2 has to execute six strategies × five seeds × three benchmarks — ninety runs.
At that volume something *will* be interrupted: a laptop sleeps, a container restarts,
a run is cancelled. If an interrupted run silently loses an episode, or re-runs one and
double-counts it, the comparison table quietly becomes wrong and there is no obvious
symptom. So durability had to come first.

Building it before the agent was deliberate. The interesting part of this project is the
agent, which is exactly why it is the wrong thing to build first — with an LLM in the
loop, every bug becomes ambiguous ("is the model bad, or is the harness broken?").
Everything in this phase is deterministic and testable, so all of it can be pinned down
before any non-determinism arrives.

## 2. What I built

A command-line optimiser with no intelligence in it: it draws random parameter values,
runs a benchmark function, records the result in Postgres, and can be killed and resumed
at any point.

| File | Responsibility |
|---|---|
| `sil_agent/agent/state.py` | Every domain model, as Pydantic v2. The vocabulary of the system. |
| `sil_agent/agent/guards.py` | `validate()` — checks proposed parameters against what the simulator declares. |
| `sil_agent/agent/loop.py` | The loop, plus the recompute-from-history functions that make resume work. |
| `sil_agent/simulators/base.py` | `Simulator` protocol and `EvalCost`. |
| `sil_agent/simulators/toy.py` | Branin, Hartmann-6, Rosenbrock, and a constrained Branin variant. |
| `sil_agent/strategies/base.py` | `Strategy` protocol — the one thing the Phase 2 ablation swaps. |
| `sil_agent/strategies/random_search.py` | Uniform random sampling. The baseline floor. |
| `sil_agent/persistence/models.py` | The `runs` and `episodes` tables. |
| `sil_agent/persistence/repo.py` | `RunRepository` — the only code that knows SQL exists. |
| `sil_agent/persistence/db.py` | Engine and session construction. |
| `sil_agent/persistence/migrations/` | Alembic environment and the initial schema migration. |
| `sil_agent/cli.py` | `run`, `resume`, `show`. |
| `tests/` | 83 tests across simulators, guards, comparison logic, persistence and resume. |
| `docker-compose.yml` | Postgres 16 and Redis 7. Redis is unused until Phase 3. |
| `pyproject.toml` | Dependencies (managed with `uv`), plus ruff, mypy and pytest configuration. |

## 3. How it works

### The loop

```
load run from database
recompute step_idx and best from the episodes table
while budget remains:
    candidate = strategy.propose(goal, history, rng_for(seed, step_idx))
    candidate = validate(candidate, parameter_space)     # clamp, coerce, or reject
    result    = simulator.run(candidate)                 # the oracle
    improved  = result.better_than(best, objective)      # computed, never opinion
    append_episode(...)                                  # <- the commit point
    save_run(...)                                        # snapshot, for convenience only
```

### Why it survives being killed

Three mechanisms, and all three are needed.

**The episode insert is the commit point.** Exactly one database statement decides whether
an episode happened. Die before it and that episode is re-run from scratch; die after it
and it is skipped. There is no moment where an episode is half-recorded, so there is no
reconciliation logic — the awkward case simply does not exist.

**Position is recomputed, never remembered.** On load, `step_idx` comes from
`max(idx) + 1` over the stored episodes and `best` comes from replaying them through the
same comparison the live loop uses. The `runs` table has columns for both, but they are a
convenience for `show`, not an input. A test deliberately corrupts that snapshot — sets
`best` to nothing and `step_idx` to 0 — and asserts the reload repairs both.

**Randomness is derived from stored values.** Each episode's generator is seeded from
`(seed, episode_index)`:

```python
rng = random.Random(f"{seed}:{idx}")
```

This one is subtle and it is the mechanism most likely to be got wrong. The obvious design
— one generator created at the start of the run — works perfectly until the process
restarts, at which point the generator restarts too and episode 163 draws what episode 0
drew. The run still completes and still looks fine; it is just no longer the run it would
have been. Deriving the generator per episode from persisted values means an interrupted
run and an uninterrupted one produce byte-identical sequences, and there is a test that
asserts exactly that.

### A worked example

`run --sim hartmann6 --episodes 400 --seed 7`, killed with `kill -9` after two seconds:

```
episodes committed at the moment of death : 163  (idx 0..162)
run status left in the database           : EXECUTING
```

Then `resume --run-id b7d5837c-...`:

```
resuming at episode 163 of 400
...
rows: 400   distinct idx: 400   min: 0   max: 399
```

163, not 0 (which would redo everything) and not 164 (which would skip the episode that
was in flight when the power went out). Episode 163 had been simulated but never recorded,
so it is correctly re-run.

### The guard

`validate()` is the piece with no job yet. Nothing in Phase 1 can propose an invalid
candidate — a uniform sampler draws inside the bounds by construction. It is built now
because from Phase 3 the proposer is an LLM, and this is the component that enforces
"the LLM proposes, deterministic code disposes". Its policy:

| Situation | Response | Why |
|---|---|---|
| Unknown parameter name | reject | A hallucinated parameter is a reasoning failure; hiding it hides the failure. |
| Missing parameter | reject | Filling in a default would invent a design decision and attribute it to the model. |
| Out of bounds | clamp | Being 5% outside a range is a near-miss worth evaluating. Recorded, not silent. |
| `3.0` for an integer | coerce | JSON has one number type. A serialisation artefact, not an error. |
| `3.7` for an integer | reject | Rounding to 4 would hide a real misunderstanding. |
| NaN or infinity | reject | NaN compares `False` against everything and would silently freeze `best` forever. |

## 4. Key decisions and trade-offs

**Benchmark functions instead of a bespoke toy problem.** Branin, Hartmann-6 and
Rosenbrock have published global optima, so solution quality can be reported as *regret* —
distance from the true answer — rather than "better than where we started". They are also
what Optuna is tested against, which makes the Phase 2 TPE comparison credible rather than
an approximation I wrote myself.

**JSONB columns instead of a column per field.** The nested shapes are defined and
validated by Pydantic and they will grow — `Evaluation` gains LLM fields in Phase 4. With
JSONB that costs no migration. The price is real: Postgres will not enforce the shape of
those columns, so Pydantic is the only thing standing between a bad write and a corrupt
row. Acceptable because every read and write goes through one repository.

**Everything immutable.** All models are `frozen=True`, so "changing" a run means
constructing a new one. It is more ceremony than mutating an attribute, and it makes Rule 1
mechanical rather than a thing to remember: state that cannot be mutated in place cannot
drift from what was persisted.

**Synchronous SQLAlchemy.** Async buys nothing for a single-process CLI loop, and it makes
every function in the call chain more complicated. FastAPI arrives in Phase 6; if it needs
async, that is a contained change behind the repository.

**A `--crash-at` development flag.** It calls `os._exit(1)` immediately before an episode
is written — the worst possible instant, after the simulation has been paid for but before
anything records it. Slightly ugly to ship a flag whose only purpose is to break things,
but it turns "kill it and see" into an automated test that runs in CI on every commit.

**What would have been easier but worse:** `Base.metadata.create_all()` instead of Alembic.
One line, works immediately, and leaves no way to change the schema on a database that
already holds rows. Migrations cost an afternoon now and save the first schema change later.

**What would be better but too expensive right now:** a Redis lock on `run_id`, so two
processes cannot write to the same run. Currently the loop detects the collision (the
episode insert reports a conflict) and stops with a clear message, which is safe but not
graceful. Phase 3 is scheduled to do it properly.

### Deviations from `TECHNICAL_DESIGN.md`

Four, each forced by something that only shows up on contact with real code:

1. **`best` is a `Best` object, not a `Candidate | None`.** Deciding whether a new result
   is an improvement needs the incumbent's objective *value*, and under Rule 2 that number
   must come from the simulator rather than be recomputed. So the winning `SimResult` is
   stored alongside the candidate.
2. **`runs` has `simulator` and `strategy` columns.** Not in the design's table sketch, but
   without them `resume --run-id X` cannot reconstruct the run — it would depend on the
   user remembering what they typed, and resuming against the wrong simulator would produce
   silent garbage. Rule 1 says the state must be sufficient, so they are part of the state.
3. **`Strategy.propose` takes an explicit `rng`.** A strategy holding its own generator is
   exactly the hidden in-memory state Rule 1 forbids. See §3.
4. **`strategies/random.py` is named `random_search.py`.** A module called `random.py`
   inside a package shadows the standard library's `random` for everything in that package.

## 5. What went wrong

**The idempotency check reported the opposite of the truth.** The very first end-to-end run
died on episode 0 with "already exists — another process is writing to this run", while the
database showed the row had just been inserted successfully. `append_episode` was deciding
whether the insert had happened by checking `result.rowcount == 1`. SQLAlchemy adds its own
`RETURNING` clause to ORM-style inserts, which makes `rowcount` unreliable. The fix is to
ask the database directly — `.returning(EpisodeRow.idx)`, and then a row came back or it
did not. Uncomfortable in hindsight: the mechanism the whole phase depends on was reporting
confidently and wrongly, and only an end-to-end run caught it.

**Pydantic silently turned `True` into `1.0`.** A guard test asserting that booleans are
rejected failed with "DID NOT RAISE". `bool` is a subclass of `int` in Python, and the
`float | int | str` union on `Candidate.params` absorbed `True` into `1.0` during model
construction — before the guard ever ran. The guard's boolean check was unreachable code
protecting against something that had already happened. The rejection had to move up a
layer, into a `mode="before"` validator on `Candidate`, which is the last point at which
the evidence still exists. A good reminder that validation has to happen at the boundary,
not behind it.

**I reformatted the design document.** `ruff format .` rewrote
`docs/TECHNICAL_DESIGN.md` — it formats Python code blocks inside Markdown, and it
collapsed the hand-aligned columns in every schema sketch in the specification this project
is built against. Restored from the original and prevented with `extend-exclude = ["*.md"]`
plus `force-exclude = true`, so that even an explicit `ruff format docs/TECHNICAL_DESIGN.md`
is now a no-op. Lesson: a formatter pointed at `.` will format things that are not code.

**The first real kill test lost its own output.** Killing a run with `kill -9` worked
exactly as designed — 163 episodes durably committed, resume continued from 163. But the
log file was empty, so the `run_id` needed to resume was gone. Python buffers stdout when
it is redirected to a file rather than a terminal, and `kill -9` flushes nothing. The
durability was perfect and completely unusable. Fixed with `flush=True` on the run header
and each episode line. The failure mode is worth remembering: the guarantee held, the
affordance for using it did not, and only an honest end-to-end test could show the
difference.

**The Hartmann-6 trap, avoided rather than hit.** The brief warned that the coefficients
are easy to get subtly wrong, so I looked up the reference implementation instead of
writing it from memory. The reference `hart6.m` at sfu.ca returns a *rescaled* variant,
`-(2.58 + outer) / 1.94`, whose minimum is nowhere near the documented −3.32237.
Transcribing it faithfully would have produced a function that looked entirely plausible,
optimised smoothly, and was not the benchmark anyone else reports numbers for. Every regret
figure in Phase 2 would have been meaningless. The test asserts the value *and* the location
of the optimum, because a wrong function can still return a plausible-looking number.

**mypy caught a rigid interface.** `Simulator.run` took `dict[str, float | int | str]`, and
the tests could not pass a `dict[str, float]` to it — `dict` is invariant in its value type,
so a narrower dictionary is not accepted even though every value in it would be. Changed to
`Mapping`, which is covariant, and which also documents that the simulator does not modify
its input.

## 6. What this unlocks

Phase 2 can now build the evaluation harness and the baselines, because the three things it
depends on are in place and tested:

- **Runs are durable**, so ninety runs can execute unattended and survive interruption.
- **Runs are reproducible**, so a difference between two strategies is a real difference and
  not a different random draw.
- **Runs are queryable**, so the ablation report is SQL over the `episodes` table rather
  than log scraping.

Specifically depending on this phase: `Strategy` (grid search, Optuna TPE and the LLM
strategies all implement the protocol `RandomSearch` already satisfies); `known_optimum` on
each benchmark, which is what makes *regret* reportable; and the guard, which in Phase 3
becomes the thing standing between a hallucinating planner and the oracle.

## Numbers

| Measurement | Value |
|---|---|
| Episodes end to end | 200 episodes in 2.62 s (~13 ms/episode) |
| Simulator cost alone | Branin 4.2 µs, Hartmann-6 10.3 µs, Rosenbrock 4.4 µs per evaluation |
| Where the 13 ms goes | Almost entirely the two database round trips per episode; the simulator is ~0.1% of it |
| Tests | 83, green, 7.2 s total |
| Code | 25 Python files, ~3,560 lines including comments and docstrings |
| Kill-and-resume, run 1 | killed at episode 163 of 400 → resumed at 163 → 400 rows, 400 distinct indices |
| Kill-and-resume, run 2 | killed at episode 152 of 500 → resumed at 152 → 500 rows, 500 distinct indices |
| LLM spend | €0.00 |

Random search quality, for Phase 2 to beat:

| Benchmark | Evaluations | Best found | True optimum | Regret |
|---|---|---|---|---|
| Branin | 200 | 0.5286 | 0.397887 | 0.131 |
| Hartmann-6 | 400 | −2.5874 | −3.32237 | 0.735 |
| Rosenbrock (4-D) | 500 | 235.63 | 0.0 | 235.6 |

Rosenbrock is the interesting one. Random search finds the valley and then has no way to
follow it, so 500 evaluations buy almost nothing. That is precisely the gap an agent that
reasons about *why* a result came out as it did is supposed to close — and it means Phase 4
has a benchmark where the answer will not be ambiguous.

## Interview angle

I can claim durable execution and demonstrate it: kill the process with `kill -9` at an
arbitrary point and the run resumes at exactly the right episode, with no loss and no
duplication, producing a sequence identical to one that was never interrupted. The evidence
is three mechanisms — a single-statement commit point, position recomputed from an
append-only table rather than trusted from a snapshot, and per-episode RNG seeding — plus
tests that assert each one, including a test that deliberately corrupts the snapshot and
checks the system repairs it from the durable record.

The sharper story is the RNG. Almost everyone building this would use one generator per
run, and it would pass every test they wrote, because the bug only appears after a restart
and only shows up as "the numbers are different from what they would have been" — which
nothing detects unless you thought to check.
