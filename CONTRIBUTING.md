# Contributing

Read `docs/TECHNICAL_DESIGN.md` before making architectural decisions.

## What this project is

An LLM agent in a closed loop with an engineering simulator. It proposes design
parameters, runs a simulation, evaluates the result against the objective,
diagnoses why it fell short, and re-plans, carrying forward what it learned.
Benchmarked against random search, grid search and Optuna TPE.

## Two rules that override convenience

**1. The loop is a pure function of persisted state.** Given `(goal, history, best)`
from the database, the next episode is fully determined. No hidden in-memory state
that the loop depends on. If something needs caching, it goes in `RunState` or is
recomputed.

**2. The LLM proposes, deterministic code disposes.** The model never computes a
metric, never decides whether a candidate improved, never invents a parameter name.
`improved`, `delta_vs_best` and `feasible` are computed from the simulator and
injected into the critic. Every LLM output is validated against a Pydantic schema
before use.

A change that would violate either rule is flagged, not worked around.

## LLM providers

All access goes through `services/router.py`. No agent code calls a provider SDK
directly.

Free tiers rate-limit at 10–15 requests per minute and agent loops hit this
constantly, so every provider adapter retries HTTP 429 with jittered exponential
backoff.

## Conventions

- Type hints everywhere; `mypy` clean; `ruff` clean.
- Protocols (`typing.Protocol`) for `Simulator`, `Strategy`, `ModelRouter`:
  swappable implementations are core to the design.
- No secrets in code: `.env` only, `.env.example` committed.
- Every experiment takes an explicit `seed`.
- Migrations for all schema changes; tables are never edited by hand.
- Tests for guards, validation, termination logic and persistence. LLM calls are
  mocked in tests; the default suite runs offline.
- Clarity over cleverness: no dense one-liners, no metaprogramming, no premature
  abstraction.

## Phase logs

Each completed phase gets `docs/phases/phase-NN.md`, following the template in
`docs/phases/README.md`. Six required sections, including "what went wrong",
which must not be empty.

## Build order

Phases 1–2 contain no LLM code. The measurement harness exists before the agent,
so every later claim about the agent is grounded in numbers fixed before there
was a result to hope for.
