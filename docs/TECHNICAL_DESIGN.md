# Technical design

The authoritative technical document. Supersedes earlier drafts.

Project name: **SIL Agent** (Simulation In the Loop).

---

## 0. Two governing rules

**Rule 1 — The loop is a pure function of persisted state.**
Given `(goal, history, best)` loaded from the database, the next episode is fully determined in structure. No hidden in-memory state. Checkpointing, resume, replay and debugging all fall out of this for free rather than being retrofitted.

**Rule 2 — The LLM proposes, deterministic code disposes.**
Every model output passes a validation guard before it touches the system. The model never computes a metric, never decides whether it improved, never invents a parameter name.

Everything below follows from these two rules.

---

## 1. Layered view

```
┌───────────────────────────────────────────────────────┐
│  API            FastAPI                                │
│                 POST /runs  GET /runs/{id}  WS /stream │
├───────────────────────────────────────────────────────┤
│  Orchestrator   Planner → Guard → Executor →           │
│                 Critic → Replanner → Checkpoint        │
├───────────────────────────────────────────────────────┤
│  Services       Memory  Budget  Router  Trace  Surrogate│
├───────────────────────────────────────────────────────┤
│  Tools (MCP)    run_simulation  describe_space         │
│                 get_history     compare_candidates     │
├───────────────────────────────────────────────────────┤
│  Simulators     Toy │ VehicleEnergy │ FMU              │
├───────────────────────────────────────────────────────┤
│  Persistence    Postgres (state + episodes)            │
│                 Redis (queue + locks)                  │
└───────────────────────────────────────────────────────┘
```

Only the Orchestrator knows about LLMs. Simulators, persistence and baselines are LLM-free — which is exactly what lets non-LLM baselines run through the identical harness.

---

## 2. State schema

```python
class RunState:
    run_id:     UUID
    goal:       Goal
    status:     Literal[PENDING, PLANNING, EXECUTING, CRITIQUING,
                        REPLANNING, DONE, FAILED, ABORTED]
    history:    list[Episode]        # append-only
    best:       Candidate | None
    budget:     BudgetState
    step_idx:   int
    seed:       int
    created_at: datetime
    updated_at: datetime
```

### Goal — parsed once by the LLM, then validated

```python
class Goal:
    raw_text:        str
    objective:       Objective          # metric name + MINIMISE | MAXIMISE
    constraints:     list[Constraint]   # metric, operator, threshold
    parameter_space: ParameterSpace     # declared by the SIMULATOR
```

```python
class ParamSpec:
    name:    str
    kind:    Literal[FLOAT, INT, CATEGORICAL]
    bounds:  tuple[float, float] | None
    choices: list[str] | None
    unit:    str | None
```

Goal parsing is the only place the LLM touches the problem specification, and its output is validated against the space the **simulator** declares. Invent a parameter that doesn't exist and you fail at episode 0, not episode 50.

### Episode — one loop iteration

```python
class Episode:
    idx:         int
    candidate:   Candidate
    result:      SimResult | ToolError
    evaluation:  Evaluation
    decision:    ReplanDecision
    cost:        CostRecord
    duration_ms: int
```

```python
class Candidate:
    params:    dict[str, float | int | str]
    rationale: str                    # why these values
    source:    Literal[PLANNER, EXPLOIT, EXPLORE, REPAIR, PERTURB, SURROGATE]
```

```python
class SimResult:
    metrics:               dict[str, float]
    objective_value:       float
    constraint_violations: list[Violation]
    feasible:              bool
    artifacts:             dict[str, str]
    wall_time_s:           float
```

### Evaluation — structured, never prose

```python
class Evaluation:
    improved:      bool       # COMPUTED, injected — not the LLM's opinion
    delta_vs_best: float      # COMPUTED, injected
    feasible:      bool       # COMPUTED, injected
    diagnosis:     str        # LLM: why this result came out this way
    hypotheses:    list[str]  # LLM: what to try next and why
    confidence:    float
```

The first three fields are computed from the oracle and passed *into* the critic. The LLM explains; it does not grade. This is Rule 2 in its most important application.

```python
class ReplanDecision:
    action:     Literal[EXPLOIT, EXPLORE, REPAIR, DECOMPOSE, ESCALATE, TERMINATE]
    reason:     str
    next_focus: list[str]
```

---

## 3. The loop

```python
while True:
    if budget.exhausted():            terminate(BUDGET);     break
    if stagnation.triggered(history): terminate(STAGNATION); break
    if goal_satisfied(best, goal):    terminate(SUCCESS);    break

    candidate = planner.propose(goal, history, best)          # LLM
    candidate = validate(candidate, goal.parameter_space)     # deterministic
    if is_duplicate(candidate, history):
        candidate = perturb(candidate, seed)

    result = simulator.run(candidate)                         # ORACLE

    improved   = result.better_than(best, goal.objective)     # deterministic
    evaluation = critic.evaluate(result, goal, best, improved)  # LLM
    decision   = replanner.decide(evaluation, history)          # LLM

    history.append(Episode(...))
    checkpoint(state)
    if improved:
        best = candidate
```

### Termination — three independent exits

| Exit | Trigger |
|---|---|
| Success | Objective target met, all constraints satisfied |
| Budget | Token, wall-clock or evaluation budget exhausted |
| Stagnation | Any detector below fires |

**Stagnation detectors** (pluggable list):

- No objective improvement in the last N episodes (default 8)
- Candidate diversity collapsed — recent proposals within ε in normalised space
- Critic confidence declining across a window
- Repeated near-identical diagnoses

Toy agents run forever or stop after a fixed N. Detecting your own lack of progress is rare and worth having.

---

## 4. Checkpointing and resume

```sql
runs     (run_id PK, status, goal JSONB, best JSONB, budget JSONB,
          step_idx, seed, created_at, updated_at)

episodes (run_id FK, idx, candidate JSONB, result JSONB,
          evaluation JSONB, decision JSONB, cost JSONB,
          duration_ms, created_at,
          PRIMARY KEY (run_id, idx))
```

`episodes` is append-only; `runs` holds the rolling snapshot.

**Resume** = load the run row, load episodes ordered by idx, continue at `step_idx`. Because of Rule 1 this needs no special-case code.

**Idempotency** — natural key `(run_id, step_idx)`, insert `ON CONFLICT DO NOTHING`. A worker crashing mid-episode cannot double-charge budget or double-run a simulation.

**Locking** — Redis lock on `run_id`, TTL exceeding max episode duration, renewed on heartbeat.

---

## 5. LLM providers — local primary, paid only for the scaling result

**Revised after Phase 3. The original version of this section was wrong** — it cited free-tier allowances from third-party aggregators that did not survive contact with reality.

Measured, not quoted:

| Provider | What was actually observed |
|---|---|
| **Cerebras** | HTTP 402 on every model. No usable free quota. |
| **Google Gemini** | `GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20`. Twenty requests per day per model — roughly 50× less than the aggregators claimed. |
| **Groq** | Untested. Worth a key as a middle tier; do not let a headline depend on it. |

Phase 4 needs roughly 2,700 calls. At 20/day that is 135 days. Free API tiers cannot carry this project.

### Local is the primary, for a better reason than cost

A pinned local model reruns **byte-identically in two years**. No hosted API can promise that — providers update models silently and without notice, and when that happens previously published numbers stop being reproducible. For a project whose entire claim is honest measurement, that is a methodological advantage, not a consolation prize.

**Target hardware: RTX 3050 Ti Laptop, 4 GB VRAM, 16 GB system RAM.** VRAM is the binding constraint.

| Model | Size at Q4_K_M | Fits 4 GB VRAM? |
|---|---|---|
| **Qwen3-4B** | ~2.5 GB | **Yes**, with headroom for KV cache |
| Qwen3-8B | ~4.7 GB | No — forces CPU offload, roughly 3× slower |

**Use `qwen3:4b-q4_K_M`.** Pin the full tag including quantisation. A bare `qwen3` tag silently changes underneath you and destroys the reproducibility this switch exists to obtain.

Qwen3-4B is also the Phase 10 distillation target, so the model used from Phase 3.5 onward is the same one that later gets fine-tuned on frontier-model traces. One story, not two.

### Structured output — the real risk at 4B

Small models are materially worse at returning schema-valid JSON. Mitigations, in order:

1. **Use Ollama's `format` parameter with a JSON schema.** This performs constrained decoding against a grammar — the model physically cannot emit invalid JSON. Do not rely on asking politely in the prompt.
2. **Measure and report the schema-compliance rate.** It belongs in the report as a number. A weakness that is measured is a result; a weakness that is hidden is a liability.
3. The router's existing one-repair-then-record-failure path stays as the backstop.

### Verify the GPU is actually being used

`ollama ps` shows whether a model is resident on GPU or CPU. A silent CPU fallback turns a four-hour overnight run into a multi-day one. Check once, at the start.

### The two published results

| Result | Model | Purpose |
|---|---|---|
| **Full ablation** | Pinned `qwen3:4b-q4_K_M` | The headline. Reproducible indefinitely. |
| **Reduced comparison** | Frontier model, paid | Does reflection help more or less as the model gets stronger? |

The second is a genuine research question, not a formality. Budget €30–80, spent once, only after the local pipeline is fully working.

### Realistic Phase 4 scope

3 LLM strategies × 5 seeds × 2 benchmarks × 20 evaluations × 3 calls ≈ **1,800 calls ≈ 4 hours** on the target hardware. An overnight run.

Five seeds is non-negotiable — Phase 2 measured the seed noise floor at ~102% of the mean. **Prefer fewer benchmarks at five seeds over more benchmarks at one.** A single-seed result is not interpretable and this project has already proven that.

### Design consequences

- The `ModelRouter` abstraction means provider choice is a config change, never a code change. This is precisely why it exists.
- Hosted free tiers rate-limit hard, and an agent loop hits that constantly. **Retry with jittered exponential backoff on HTTP 429 is mandatory.** Already built in Phase 3.
- Keep the Gemini adapter. Cross-provider failover is portfolio content in its own right.

```python
class ModelRouter(Protocol):
    def complete(self, role: Role, prompt: Prompt,
                 schema: type[BaseModel]) -> tuple[BaseModel, CostRecord]: ...
```

| Role | Tier | Why |
|---|---|---|
| Goal parsing | Strong | Runs once; wrong here poisons everything |
| Planner | Strong | Reasoning-heavy |
| Critic | Strong | Diagnosis quality *is* the value of the loop |
| Replanner | Mid | Structured decision over critic output |
| Summarise / format | Cheap | High volume, low stakes |

---

## 6. Budget governor

Three independent budgets; any one exhausted halts the run.

| Budget | Typical | Notes |
|---|---|---|
| Token / euro | €2.00 per run | Guards LLM spend |
| Wall-clock | 30 min | Guards hangs |
| Evaluations | 100 sim calls | Usually the real constraint |

**Reserve-then-commit** — reserve estimated cost before each call, commit actual after, release the difference. Fail closed. Check before *every* call, not just at loop top; one episode makes 3+ LLM calls. Enforce per-call token caps to catch runaway generation.

The evaluation budget is what makes the ablation fair: every strategy gets exactly the same number of simulator calls.

---

## 7. Trace format

OpenTelemetry spans exported to Langfuse. Root span per run, child per episode, grandchild per component call.

```
run  {run_id, goal, seed}
└── episode[3]  {objective.best_so_far}
    ├── planner.propose   {model, tokens_in/out, cost_eur, latency_ms}
    ├── guard.validate    {passed, clamped_params}
    ├── simulator.run     {params, wall_time_s, objective_value, feasible}
    ├── critic.evaluate   {model, improved, confidence}
    └── replanner.decide  {model, action, next_focus}
```

Custom attributes worth setting: `agent.episode.idx`, `agent.objective.value`, `agent.objective.best_so_far`, `agent.decision.action`, `agent.budget.remaining_evals`, `agent.stagnation.score`.

With these you can plot objective-vs-episode against cost-vs-episode and actually watch the agent reason.

---

## 8. Simulator interface

```python
class Simulator(Protocol):
    def describe(self) -> ParameterSpace: ...
    def run(self, params: dict) -> SimResult: ...
    @property
    def cost_per_eval(self) -> EvalCost: ...
```

| Implementation | Purpose |
|---|---|
| `ToySimulator` | Analytic, instant. Develop and debug against this. |
| `VehicleEnergySimulator` | Longitudinal dynamics + energy + simple thermal, pure Python. |
| `FMUSimulator` | Wraps FMPy — lets you plug in a real Simulink export later. |

**Make `ToySimulator` wrap standard benchmark functions — Branin, Hartmann-6, constrained Rosenbrock.** This is the highest-leverage practical decision in the build:

- Known global optima → measure true regret, not just relative improvement
- Optuna handles them natively → free, credible Bayesian baselines
- Instant evaluation → hundreds of iterations while debugging, at zero cost
- Standard in the literature → your numbers are comparable to published work

---

## 9. Baselines — build these before the agent

```python
class Strategy(Protocol):
    def propose(self, goal: Goal, history: list[Episode]) -> Candidate: ...
```

| Strategy | Tests |
|---|---|
| `RandomSearch` | The floor |
| `GridSearch` | Naive systematic coverage |
| `OptunaTPE` | The honest competitor |
| `SingleShotLLM` | Does looping help at all? |
| `AgentNoReflection` | Planner only, critic disabled |
| `AgentFull` | Planner + critic + memory |

Same simulator, same budget, same harness for all six.

**Seeds matter.** LLMs are stochastic — run ≥5 seeds per configuration and report mean ± standard deviation. Single-run comparisons are noise, and a sharp interviewer will ask.

---

## 10. MCP surface

```
start_run(goal_text)            → run_id
run_simulation(params)          → SimResult
describe_parameter_space()      → ParameterSpace
get_run_history(run_id)         → list[Episode]
compare_candidates(a, b)        → Comparison
```

`describe_parameter_space` is the schema `validate()` checks against — the simulator is the authority on what exists, never the model. Payoff: Claude Desktop or Claude Code can drive your optimiser directly.

---

## 11. Depth extensions

These are what take the project from good to unusual. Each adds a distinct skill area without a second codebase.

### 11.1 Surrogate model (highest value — plays to your ML background)

Train a cheap model (Gaussian Process, or a small NN) on completed evaluations to *predict* simulator output. Use it to pre-screen the agent's proposals: generate five candidates, predict all five with the surrogate, run the real simulator only on the most promising.

Why it's strong: it puts your actual ML expertise into an LLM project, it's a real technique from Bayesian optimisation, and it produces a headline result — *"cut expensive simulator calls by 60% at equal solution quality."* Cost engineering that isn't just token counting.

### 11.2 Distil the planner into a local model

Collect successful planning traces from a strong model, then QLoRA fine-tune a small open model (Qwen3-4B) to reproduce them. Measure quality retained versus cost.

Why it's strong: adds fine-tuning and distillation to the project, gives you free inference forever, and **merges this project with your local-models project** — one story instead of two. *"I distilled my planner into a 4B model running on my laptop at 3% of the cost, retaining 91% of solution quality."*

### 11.3 Multi-objective optimisation (Pareto fronts)

Real engineering is never single-objective — it's cost vs performance vs weight vs efficiency. Return a Pareto front rather than one answer, and have the agent reason about trade-offs.

Why it's strong: domain authenticity. Any mechanical engineer interviewing you will recognise immediately that you understand real design work.

### 11.4 Prompt versioning and A/B testing

Treat prompts as versioned artifacts. Run the eval harness across prompt versions and keep the results. Gate prompt changes in CI the same way you gate code.

Why it's strong: almost nobody does this, and it's a genuine production practice. *"My prompts are version-controlled and every change is A/B tested against a regression suite."*

### 11.5 Human-in-the-loop escalation

Implement the `ESCALATE` action properly — when the agent detects stagnation it pauses, surfaces its state and hypotheses, and waits for human input, then resumes with that input in context.

Why it's strong: completes the durable-execution story (you already have the checkpointing to support it) and shows you think about where humans belong in autonomous systems.

---

## 12. Repo layout

```
sil_agent/
  agent/        loop.py  planner.py  critic.py  replanner.py
                guards.py  stagnation.py  state.py
  simulators/   base.py  toy.py  vehicle.py  fmu.py
  strategies/   random.py  grid.py  optuna_tpe.py  llm_single.py
  services/     budget.py  memory.py  router.py  trace.py  surrogate.py
  mcp/          server.py  tools.py
  api/          main.py  routes.py  schemas.py
  persistence/  models.py  repo.py  migrations/
  eval/         harness.py  ablation.py  report.py
  prompts/      versioned prompt templates
  tests/
docs/
  WHY_THIS_PROJECT.md
  TECHNICAL_DESIGN.md
  phases/       phase-01.md, phase-02.md, ...
docker-compose.yml
.github/workflows/ci.yml
```

---

## 13. Phase plan

### Core — a complete, strong portfolio project on its own (~6–8 weeks)

| Phase | Deliverable | Proves |
|---|---|---|
| **1** | State schema, Postgres, `ToySimulator`. **No LLM.** | Checkpoint/resume works — kill it mid-run, resume clean |
| **2** | Baselines + eval harness + ablation report | You can measure anything you build from here |
| **3** | Agent loop, planner only. Retry/backoff on 429. | Loop runs end to end on free tiers |
| **4** | **Critic + replanner. Re-run ablation.** | **The money experiment: does reflection pay?** |
| **5** | Episodic memory. Re-run ablation. | Does not-repeating-yourself help? |
| **6** | Budget governor, OTel tracing, MCP server, FastAPI | Production spine |

### Depth — what makes it unusual (~4 weeks)

| Phase | Deliverable |
|---|---|
| **7** | Surrogate model pre-screening |
| **8** | Multi-objective / Pareto fronts |
| **9** | `VehicleEnergySimulator` — domain credibility |
| **10** | Distil planner into local Qwen3-4B |

### Polish (~1 week)

| Phase | Deliverable |
|---|---|
| **11** | Prompt versioning + A/B, human-in-the-loop escalation |
| **12** | Blog post, demo video, README, architecture diagram |

**Phases 1–2 involve no LLM at all.** That's deliberate — you'll have a rigorous measurement harness before you have an agent, so every later claim is grounded.

**Phase 4 is the experiment the project exists to run.** Whatever it shows, including a negative result, is publishable because you measured it honestly.

**Ship after Phase 6.** Put it on the CV, start talking about it, then keep extending. Don't wait for Phase 12 to tell anyone it exists.
