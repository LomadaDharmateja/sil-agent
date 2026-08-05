# Phase 1 brief — state, persistence, toy simulator

Written **before** building. The log (`phase-01.md`) gets written after.

**No LLM code in this phase.** None. If you find yourself importing an LLM SDK, you're in the wrong phase.

---

## Goal of this phase

Prove **durable execution**: a run can be killed at any point and resumed with no loss and no duplication.

Everything later — hundred-step agent runs, expensive simulations, budget tracking — depends on this being solid. Building it first, without the distraction of an LLM, means it gets built properly.

---

## What to build

### 1. Domain models (`agent/state.py`)

Pydantic v2 models exactly as specified in `docs/TECHNICAL_DESIGN.md` §2:

`ParamSpec`, `ParameterSpace`, `Objective`, `Constraint`, `Goal`, `Candidate`, `Violation`, `SimResult`, `Evaluation`, `ReplanDecision`, `CostRecord`, `BudgetState`, `Episode`, `RunState`

For this phase `Evaluation` and `ReplanDecision` can be constructed with placeholder values — they're populated by the LLM in phases 3–4. Define them now so the schema is stable.

### 2. Simulator protocol and toy implementations (`simulators/`)

```python
class Simulator(Protocol):
    def describe(self) -> ParameterSpace: ...
    def run(self, params: dict) -> SimResult: ...
    @property
    def cost_per_eval(self) -> EvalCost: ...
```

Implement `ToySimulator` wrapping three standard benchmark functions:

| Function | Domain | Known global minimum |
|---|---|---|
| Branin | x1 ∈ [-5, 10], x2 ∈ [0, 15] | **0.397887**, at three separate points |
| Hartmann-6 | [0, 1]⁶ | approximately **-3.32237** |
| Rosenbrock (constrained) | typically [-5, 10]ⁿ | **0**, at all xᵢ = 1 |

Look up the exact coefficient matrices rather than trusting memory — Hartmann-6 in particular has a 4×6 alpha/A/P coefficient set that is easy to get subtly wrong.

Add a constraint variant so constraint handling is exercised from day one: e.g. Branin with `x1 + x2 <= 10`.

### 3. Persistence (`persistence/`)

Postgres, two tables per `TECHNICAL_DESIGN.md` §4.

- SQLAlchemy models plus Alembic migrations. **Never create tables by hand** — migrations are how schema changes stay reproducible across machines.
- Repository layer: `save_run`, `load_run`, `append_episode`, `load_episodes`
- `append_episode` must be idempotent on `(run_id, idx)` using `ON CONFLICT DO NOTHING`

### 4. A trivial strategy (`strategies/random.py`)

Uniform random sampling within the parameter space. No intelligence — it exists purely to drive the loop so persistence can be tested end to end.

### 5. Loop skeleton (`agent/loop.py`)

The structure from `TECHNICAL_DESIGN.md` §3, with LLM steps stubbed:

```
load or create run
while not done:
    candidate = strategy.propose(goal, history)
    candidate = validate(candidate, goal.parameter_space)
    result    = simulator.run(candidate)
    improved  = result.better_than(best, goal.objective)
    append_episode(...)
    checkpoint(state)
    if improved: best = candidate
```

`validate()` is real in this phase — clamp out-of-bounds values, reject unknown parameter names, coerce types. It's the guard that later stops the LLM inventing parameters, so build it properly now.

### 6. CLI (`cli.py`)

```
python -m sil_agent.cli run --sim branin --episodes 20 --seed 42
python -m sil_agent.cli resume --run-id <uuid>
python -m sil_agent.cli show --run-id <uuid>
```

### 7. Infrastructure

`docker-compose.yml` with Postgres and Redis. `.env.example` committed, `.env` ignored.

### 8. Tests (`tests/`)

- Simulator correctness — each benchmark returns its known optimum at its known optimal point. **If Branin doesn't evaluate to 0.397887 at (π, 2.275), the implementation is wrong.**
- `validate()` clamps, rejects unknown names, coerces types
- `append_episode` is idempotent — call twice, get one row
- Resume produces an identical continuation given the same seed

---

## Acceptance criteria

Phase 1 is done when all of these hold:

1. `run --episodes 20` completes and persists exactly 20 episode rows
2. Killing the process at roughly episode 10 and running `resume` continues from 10 — not 0, and not 11
3. Calling `append_episode` twice with the same `(run_id, idx)` leaves one row
4. Each benchmark function returns its documented optimum at its documented optimal point
5. Two runs with the same seed produce identical episode sequences
6. `ruff` clean, `mypy` clean, `pytest` green

Criterion 2 is the one that matters. Actually do it — `Ctrl-C` a real run, then resume it.

---

## What this unlocks

Phase 2 (baselines and the eval harness) needs to run six strategies × five seeds × three benchmarks = 90 runs and compare them. That's only possible if runs are durable, seeded and queryable — which is exactly what this phase delivers.

---

## Explicitly out of scope

- Any LLM call
- Budget governor (Phase 6)
- Tracing and observability (Phase 6)
- FastAPI and MCP (Phase 6)
- Stagnation detection (Phase 3)
- Surrogate models (Phase 7)

Resist all of it. The value of this phase is that it's small enough to get exactly right.

---

## Finally

Write `docs/phases/phase-01.md` using the template in `README.md`. All six sections, including "what went wrong."
