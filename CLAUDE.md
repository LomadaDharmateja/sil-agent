# SIL Agent — project context

Standing instructions for Claude Code in this repo. Read `docs/TECHNICAL_DESIGN.md` before making architectural decisions.

## What this project is

An LLM agent in a closed loop with an engineering simulator. It proposes design parameters, runs a simulation, evaluates the result against the objective, diagnoses why it fell short, and re-plans — carrying forward what it learned. Benchmarked against random search, grid search and Optuna TPE.

Purpose: portfolio project for AI engineering roles in Germany. Optimised for demonstrating production engineering rigour, not for shipping a product.

## Two rules that override convenience

**1. The loop is a pure function of persisted state.** Given `(goal, history, best)` from the database, the next episode is fully determined. Never introduce hidden in-memory state that the loop depends on. If you're tempted to cache something on an object, put it in `RunState` or recompute it.

**2. The LLM proposes, deterministic code disposes.** The model never computes a metric, never decides whether a candidate improved, never invents a parameter name. `improved`, `delta_vs_best` and `feasible` are computed from the simulator and injected into the critic. Every LLM output is validated against a Pydantic schema before use.

If a change would violate either rule, stop and flag it rather than working around it.

## The user

Strong background in ML, deep learning and mechanical engineering. Newer to software architecture and production engineering patterns — this project is partly how he learns them.

**So: explain as you go.** When introducing a pattern (idempotency, dependency injection, protocols, migrations, connection pooling), add one or two plain sentences on what it is and why it's used here. Don't assume the vocabulary. Don't skip it either — the learning is a goal, not overhead.

Prefer clarity over cleverness in all code. No dense one-liners, no metaprogramming, no premature abstraction.

## Phase logs — required

After completing each phase, write `docs/phases/phase-NN.md` following the template in `docs/phases/README.md`. Six required sections, including "what went wrong", which must not be empty.

Do this as part of the phase, not as an afterthought.

## Stack

- Python 3.12, Pydantic v2 for all schemas
- Postgres (state, episodes), Redis (queue, locks)
- FastAPI, Docker Compose
- Optuna for baselines, OpenTelemetry + Langfuse for tracing
- pytest, ruff, mypy

## LLM providers

Free tiers during development — Cerebras, Gemini, Groq, NVIDIA NIM, OpenRouter. Paid frontier models only for final benchmark runs.

All access goes through `services/router.py`. Never call a provider SDK directly from agent code.

Free tiers rate-limit at 10–15 RPM and agent loops hit this constantly. Retry with jittered exponential backoff on HTTP 429 is required in every provider adapter.

## Conventions

- Type hints everywhere; `mypy` clean
- Protocols (`typing.Protocol`) for `Simulator`, `Strategy`, `ModelRouter` — swappable implementations are core to the design
- No secrets in code; `.env` only, `.env.example` committed
- Every experiment takes an explicit `seed`
- Migrations for all schema changes — never edit tables by hand
- Tests for guards, validation, termination logic and persistence. LLM calls are mocked in tests.

## Build order

Phases 1–2 contain no LLM code at all. This is deliberate: the measurement harness exists before the agent, so every later claim about the agent is grounded. Do not skip ahead to the agent because it's more interesting.
