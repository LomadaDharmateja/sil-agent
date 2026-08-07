# SIL Agent — Simulation In the Loop

An LLM agent that optimises engineering designs by reasoning in a closed loop with a simulator, benchmarked honestly against classical optimisation.

> Give it a goal — *"minimise energy consumption, keep 0–100 km/h under 9 s"* — and it proposes settings, runs the simulation, works out why the result fell short, and tries again with what it learned. Every number is computed by the simulator; the model only proposes and explains.

---

## The problem

Finding the best settings for an engineering design means searching a space of billions of combinations where each evaluation costs minutes of simulation time. Today you either have an engineer do it by hand — good judgement, no throughput, no written record of the reasoning — or an optimiser do it — high throughput, no domain knowledge, no explanation.

This closes that gap: the throughput of an optimiser, reasoning that can use knowledge expressed in language, and a readable justification for every decision.

## Why it isn't just an LLM guessing

The model never grades itself. Whether a candidate improved is computed from the simulator's objective value and injected into the critic. Every proposed parameter is validated against a schema the simulator declares. The LLM proposes and explains; **reality decides.**

## Status

Under construction. See `docs/phases/` for the build log.

| Phase | Status |
|---|---|
| 1 — State, persistence, toy simulator | **done** — [log](docs/phases/phase-01.md) |
| 2 — Baselines and eval harness | **done** — [log](docs/phases/phase-02.md) · [report](reports/phase2-main/report.md) |
| 3 — Agent loop, planner only | **done** — [log](docs/phases/phase-03.md) · [report](reports/phase3-main/report.md) |
| 3.5 — Memorisation fix, local provider | **done** — [log](docs/phases/phase-035.md) · [report](reports/phase35-main/report.md) |
| 4 — Critic and replanner | not started |
| 5 — Episodic memory | not started |
| 6 — Budget, tracing, MCP, API | not started |
| 7–10 — Surrogate, Pareto, vehicle sim, distillation | not started |
| 11–12 — Prompt versioning, HITL, write-up | not started |

## The baseline to beat

Phases 1–2 contain no LLM code. The measurement harness exists before the agent,
so every later claim about the agent is grounded in numbers that were fixed
before there was a result to hope for.

Final regret over 5 seeds, 200 simulator calls per run, lower is better:

| Strategy | Branin | Hartmann-6 | Rosenbrock (4-D) |
|---|---|---|---|
| Grid search | 0.0204 | 3.157 | 4226 |
| Random search | 0.153 ± 0.12 | 0.866 ± 0.42 | 547 ± 830 |
| **Optuna TPE** | **0.0188 ± 0.024** | **0.152 ± 0.10** | **28.9 ± 33** |

TPE is the number to beat — **at 200 evaluations**, which is the regime it is
built for and where the agent has not been measured. Any comparison has to name
its budget, because these strategies do not rank the same at 20 as at 200.

The [full report](reports/phase2-main/report.md) carries per-seed data,
convergence plots and a measured **seed noise floor** — how far apart two
five-seed means can be when nothing differs but the seeds, which is the bar any
Phase 4 claim has to clear.

```bash
python -m sil_agent.cli ablate --experiment phase2-main --seeds 5 --episodes 200
python -m sil_agent.cli report --experiment phase2-main --noise-experiment phase2-noise
```

Re-running `ablate` with the same arguments continues an interrupted matrix — run
identity is derived from the configuration, so there is no resume flag to remember.

## How the agent is kept honest

The LLM is fenced in by deterministic code at every point where it could quietly
corrupt a result:

| Boundary | What it does |
|---|---|
| **`services/router.py`** | The only door to a provider. Model output becomes an object solely by surviving `json.loads` and Pydantic validation — one repair attempt, then a recorded failure. |
| **`agent/guards.py`** | A proposed parameter is checked against the space the *simulator* declares. Invented names are rejected, out-of-bounds values clamped and the clamp recorded. |
| **Duplicate detection** | A planner shown a good result will re-propose it indefinitely, with a fluent justification each time. Repeats are perturbed deterministically and marked `PERTURB` so the rate stays visible. |
| **Budget accounting** | Rejections are counted and capped separately from evaluations, so a hallucinating planner still gets the same number of *simulator calls* as a well-behaved one. |
| **`llm_calls` table** | Every prompt and reply is stored, so a run replays offline and the evidence behind every published number survives. |
| **Shifted instances** | Benchmarks are re-posed on a seeded rotation of themselves, so a model that memorised the textbook cannot recall the answer. A test asserts nothing in the prompt names the function. |

Rate limiting never reaches agent code: the router paces calls to the provider's
limit and retries 429s with jittered backoff.

### The result Phase 3 did not claim, and what replaced it

On Branin at 15 evaluations both LLM strategies reached the global optimum while
TPE sat at regret 4.9 — a seven-order-of-magnitude win that **should not have been
believed.** The single-shot control proposed all three of Branin's global minima
to four decimal places *before seeing any result*: the model had memorised the
benchmark, so the comparison measured recall rather than search.

Phase 3.5 fixed the measurement. Every benchmark is now also published as a
**shifted and rotated instance** — `f'(z) = f(R(z − s))`, seeded, posed on the
unit cube with the optimal *value* preserved so regret stays reportable — and
nothing in the prompt names the function. On those instances the same control
opens with the four corners and the centre of the unit square instead of the
published minima. It is searching, not reciting.

The comparison that replaces it, on a **local 4B model** at **20 evaluations**:

| Strategy | branin_i1 | hartmann6_i1 |
|---|---|---|
| **agent_no_reflection** | **0.381 ± 0.23** | **1.052 ± 0.56** |
| optuna_tpe | 2.838 ± 2.5 | 1.540 ± 0.64 |
| single_shot_llm | 4.236 ± 2.5 | 2.347 ± 0.58 |
| random_search | 4.689 ± 3.4 | 1.684 ± 0.78 |

The agent and its no-loop control have **separated** — in Phase 3 they were
identical, because memorisation had saturated both. That gap is the loop.

### What this is not

**This is not "the agent beats TPE".** It is a claim about *sample efficiency
under a tight budget*, and the budget is the whole point. Given 200 evaluations
TPE reaches regret 0.0188 — an order of magnitude below anything the agent has
been measured at. A Bayesian optimiser spends its early evaluations building a
surrogate and only then exploits it; judging it at 20 evaluations is judging it
before it has started.

The [budget sweep](reports/phase35-sweep/sweep.md) runs both at 10, 20, 40 and 80
evaluations with a rank test at each. The agent's lead is statistically supported
only at 20 (p=0.016); by 40 the two distributions overlap (p=0.69). The medians
cross near 37, but **no budget above the crossing shows a real difference** — so
the finding is that the agent's edge *dissolves*, not that TPE overtakes. That
sweep, not the single 20-evaluation column, is the honest summary.

Two further limits are on the record rather than buried: on `hartmann6_i1` the
lead over TPE sits inside the measured seed noise floor, so **no claim is made
there**; and Optuna's `n_startup_trials` is 10, so at a 20-evaluation budget half
of TPE's run is random sampling — stated in the report, and measured rather than
assumed (lowering it makes TPE *worse*). Details in the
[Phase 3.5 log](docs/phases/phase-035.md).

## Documentation

| Document | What it covers |
|---|---|
| [`docs/WHY_THIS_PROJECT.md`](docs/WHY_THIS_PROJECT.md) | The problem, the gap, who cares, how to explain it |
| [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md) | Architecture, state schema, phase plan |
| [`docs/phases/`](docs/phases/) | Build log — one file per phase |

## Stack

Python 3.12 · Pydantic v2 · Postgres · Redis · FastAPI · Optuna · Ollama · OpenTelemetry · Langfuse · Docker

The agent runs on a **local** `qwen3:4b-q4_K_M` through Ollama — no API key, no
quota, and a pinned model that reruns byte-identically in two years, which no
hosted API can promise. Hosted providers remain wired in as failover.
