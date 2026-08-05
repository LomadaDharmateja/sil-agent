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
| 3 — Agent loop, planner only | not started |
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

TPE is the number the agent has to beat. The [full report](reports/phase2-main/report.md)
carries per-seed data, convergence plots and a measured **seed noise floor** — how far
apart two five-seed means can be when nothing differs but the seeds, which is the bar
any Phase 4 claim has to clear.

```bash
python -m sil_agent.cli ablate --experiment phase2-main --seeds 5 --episodes 200
python -m sil_agent.cli report --experiment phase2-main --noise-experiment phase2-noise
```

Re-running `ablate` with the same arguments continues an interrupted matrix — run
identity is derived from the configuration, so there is no resume flag to remember.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/WHY_THIS_PROJECT.md`](docs/WHY_THIS_PROJECT.md) | The problem, the gap, who cares, how to explain it |
| [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md) | Architecture, state schema, phase plan |
| [`docs/phases/`](docs/phases/) | Build log — one file per phase |

## Stack

Python 3.12 · Pydantic v2 · Postgres · Redis · FastAPI · Optuna · OpenTelemetry · Langfuse · Docker
