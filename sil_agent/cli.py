"""Command line interface.

    python -m sil_agent.cli run     --sim branin --episodes 20 --seed 42
    python -m sil_agent.cli resume  --run-id <uuid>
    python -m sil_agent.cli show    --run-id <uuid>
    python -m sil_agent.cli ablate  --experiment phase2-main
    python -m sil_agent.cli report  --experiment phase2-main

``argparse`` rather than a CLI framework: a handful of subcommands with a few
flags each need no decorators, no introspection and no extra dependency.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

from sil_agent.agent.loop import LoopResult, rehydrate, run_loop
from sil_agent.agent.state import (
    BudgetState,
    Episode,
    RunState,
    RunStatus,
    SimResult,
    utcnow,
)
from sil_agent.eval import report as report_module
from sil_agent.eval.harness import ExperimentSpec, execute
from sil_agent.persistence.db import make_session_factory, resolve_database_url
from sil_agent.persistence.repo import RunRepository
from sil_agent.services.replay import CachingRouter
from sil_agent.services.router import ModelRouter, build_default_router
from sil_agent.simulators.toy import BENCHMARKS, ToySimulator
from sil_agent.strategies.base import Strategy
from sil_agent.strategies.registry import STRATEGY_NAMES
from sil_agent.strategies.registry import build_strategy as _build_strategy


def build_strategy(
    name: str,
    *,
    max_evaluations: int,
    run_id: UUID,
    repo: RunRepository,
) -> Strategy:
    """Construct a strategy, turning an unknown name into a clean CLI exit.

    The budget is passed through because grid search sizes its grid from it.
    On resume it comes from the stored ``BudgetState``, never from the command
    line — see ``strategies/registry.py``.
    """

    def router_factory() -> ModelRouter:
        return CachingRouter(build_default_router(), repo, run_id)

    try:
        return _build_strategy(
            name,
            max_evaluations=max_evaluations,
            run_id=run_id,
            router_factory=router_factory,
        )
    except KeyError as exc:
        raise SystemExit(str(exc.args[0])) from None


def build_repository() -> RunRepository:
    return RunRepository(make_session_factory(resolve_database_url()))


def _print_episode(state: RunState, episode: Episode) -> None:
    # flush=True throughout this module: Python buffers stdout when it is
    # redirected to a file rather than a terminal, and a killed process never
    # flushes. That was not theoretical — the first real kill-and-resume test
    # lost the run_id line, which is the one thing needed in order to resume.
    result = episode.sim_result
    if result is None:
        assert not isinstance(episode.result, SimResult)
        print(
            f"  ep {episode.idx:>3}  FAILED  {episode.result.kind}: {episode.result.message}",
            flush=True,
        )
        return

    flags = "feasible" if result.feasible else "INFEASIBLE"
    marker = " *improved*" if episode.evaluation.improved else ""
    best_value = state.best.result.objective_value if state.best else float("nan")
    print(
        f"  ep {episode.idx:>3}  objective={result.objective_value:>12.6f}  "
        f"{flags:<10}  best={best_value:>12.6f}{marker}",
        flush=True,
    )


def _report(result: LoopResult, repo: RunRepository) -> None:
    state = result.state
    stored = repo.count_episodes(state.run_id)
    print()
    print(f"run {state.run_id}")
    print(f"  terminated:      {result.reason.value}")
    print(f"  episodes run:    {result.episodes_run} this invocation")
    print(f"  episodes stored: {stored}")
    if state.best is not None:
        print(f"  best objective:  {state.best.result.objective_value:.6f}")
        print(f"  best params:     {state.best.candidate.params}")
        print(f"  found at:        episode {state.best.episode_idx}")
    else:
        print("  best objective:  none (no successful episode)")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    simulator = ToySimulator.from_name(args.sim)
    repo = build_repository()

    run_id = uuid4()
    strategy = build_strategy(
        args.strategy, max_evaluations=args.episodes, run_id=run_id, repo=repo
    )

    now = utcnow()
    state = RunState(
        run_id=run_id,
        goal=simulator.default_goal(),
        status=RunStatus.PENDING,
        history=[],
        best=None,
        budget=BudgetState(max_evaluations=args.episodes),
        step_idx=0,
        seed=args.seed,
        created_at=now,
        updated_at=now,
    )

    # The run row must exist before any episode: episodes reference it by
    # foreign key. It is also what makes the run resumable from the very first
    # instant — printed below so it can be killed and resumed immediately.
    repo.save_run(state, simulator=simulator.name, strategy=strategy.name)

    # Printed before the first episode and flushed immediately: if this run is
    # killed, this is the identifier needed to resume it.
    print(f"run_id: {state.run_id}", flush=True)
    print(f"simulator: {simulator.name}   strategy: {strategy.name}   seed: {args.seed}")
    print(f"goal: {state.goal.raw_text}")
    print(flush=True)

    result = run_loop(
        state=state,
        simulator=simulator,
        strategy=strategy,
        repo=repo,
        on_episode=_print_episode,
        crash_at=args.crash_at,
    )
    _report(result, repo)
    return 0


def command_resume(args: argparse.Namespace) -> int:
    repo = build_repository()
    stored = repo.load_run(args.run_id)
    if stored is None:
        print(f"no run with id {args.run_id}", file=sys.stderr)
        return 1

    simulator = ToySimulator.from_name(stored.simulator)

    # rehydrate() is what makes this honest: the position comes from the
    # episodes table, not from whatever runs.step_idx happens to say.
    state = rehydrate(stored.state)

    # The budget comes from the stored state, not from a flag. Rebuilding grid
    # search with a different `max_evaluations` would give it a different grid,
    # and the resumed run would stop continuing the original one.
    strategy = build_strategy(
        stored.strategy,
        max_evaluations=state.budget.max_evaluations,
        run_id=state.run_id,
        repo=repo,
    )

    print(f"run_id: {state.run_id}", flush=True)
    print(f"simulator: {simulator.name}   strategy: {strategy.name}   seed: {state.seed}")
    print(f"resuming at episode {state.step_idx} of {state.budget.max_evaluations}")
    print(flush=True)

    result = run_loop(
        state=state,
        simulator=simulator,
        strategy=strategy,
        repo=repo,
        on_episode=_print_episode,
        crash_at=args.crash_at,
    )
    _report(result, repo)
    return 0


def command_show(args: argparse.Namespace) -> int:
    repo = build_repository()
    stored = repo.load_run(args.run_id)
    if stored is None:
        print(f"no run with id {args.run_id}", file=sys.stderr)
        return 1

    state = stored.state
    rehydrated = rehydrate(state)

    print(f"run_id:     {state.run_id}")
    print(f"status:     {state.status.value}")
    print(f"simulator:  {stored.simulator}")
    print(f"strategy:   {stored.strategy}")
    print(f"seed:       {state.seed}")
    print(f"goal:       {state.goal.raw_text}")
    print(f"objective:  {state.goal.objective.metric} ({state.goal.objective.direction.value})")
    if state.goal.constraints:
        for constraint in state.goal.constraints:
            print(
                f"constraint: {constraint.metric} {constraint.operator.value} "
                f"{constraint.threshold}"
            )
    print(f"episodes:   {len(state.history)} of {state.budget.max_evaluations}")
    print(f"step_idx:   stored={state.step_idx}  recomputed={rehydrated.step_idx}")

    if rehydrated.best is not None:
        best = rehydrated.best
        print(f"best:       {best.result.objective_value:.6f} at episode {best.episode_idx}")
        print(f"params:     {best.candidate.params}")
    else:
        print("best:       none")

    if args.episodes:
        print()
        for episode in state.history:
            _print_episode(rehydrated, episode)

    return 0


def command_ablate(args: argparse.Namespace) -> int:
    repo = build_repository()
    spec = ExperimentSpec(
        name=args.experiment,
        strategies=args.strategies,
        simulators=args.sims,
        seeds=list(range(1, args.seeds + 1)),
        max_evaluations=args.episodes,
    )

    print(f"experiment: {spec.name}")
    print(f"strategies: {', '.join(spec.strategies)}")
    print(f"benchmarks: {', '.join(spec.simulators)}")
    print(f"seeds:      1..{args.seeds}")
    print(f"budget:     {spec.max_evaluations} evaluations per run")
    print(f"total:      {spec.total_runs} runs")
    print(flush=True)

    outcomes = execute(spec, repo)

    skipped = sum(1 for o in outcomes if o.status == "skipped")
    episodes = sum(o.episodes_run for o in outcomes)
    print()
    print(f"{len(outcomes)} runs: {skipped} already complete, {len(outcomes) - skipped} executed")
    print(f"{episodes} episodes run this invocation")
    print(f"\nnext: python -m sil_agent.cli report --experiment {spec.name}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    repo = build_repository()
    out_dir = Path(args.out) if args.out else Path("reports") / args.experiment

    try:
        result = report_module.build(
            args.experiment,
            repo,
            out_dir,
            noise_experiment=args.noise_experiment,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote {result.markdown}")
    for plot in result.plots:
        print(f"wrote {plot}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sil-agent",
        description="SIL Agent — simulation-in-the-loop optimisation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="start a new run")
    run_parser.add_argument(
        "--sim",
        required=True,
        choices=sorted(BENCHMARKS),
        help="which benchmark simulator to optimise",
    )
    run_parser.add_argument("--episodes", type=int, default=20, help="evaluation budget")
    run_parser.add_argument("--seed", type=int, default=42, help="random seed")
    run_parser.add_argument("--strategy", default="random_search", choices=STRATEGY_NAMES)
    run_parser.add_argument(
        "--crash-at",
        type=int,
        default=None,
        help="development flag: kill the process just before writing this episode",
    )
    run_parser.set_defaults(func=command_run)

    resume_parser = subparsers.add_parser("resume", help="continue an existing run")
    resume_parser.add_argument("--run-id", type=UUID, required=True)
    resume_parser.add_argument("--crash-at", type=int, default=None, help=argparse.SUPPRESS)
    resume_parser.set_defaults(func=command_resume)

    show_parser = subparsers.add_parser("show", help="inspect a run")
    show_parser.add_argument("--run-id", type=UUID, required=True)
    show_parser.add_argument(
        "--episodes", action="store_true", help="list every episode, not just the summary"
    )
    show_parser.set_defaults(func=command_show)

    ablate_parser = subparsers.add_parser(
        "ablate",
        help="run a full strategy x benchmark x seed matrix",
        description=(
            "Runs every combination and stores the results under one experiment name. "
            "Safe to re-run: run identity is derived from the configuration, so an "
            "interrupted matrix is continued by issuing the identical command again."
        ),
    )
    ablate_parser.add_argument("--experiment", required=True, help="name grouping these runs")
    ablate_parser.add_argument(
        "--strategies",
        nargs="+",
        default=STRATEGY_NAMES,
        choices=STRATEGY_NAMES,
    )
    ablate_parser.add_argument(
        "--sims",
        nargs="+",
        default=sorted(BENCHMARKS),
        choices=sorted(BENCHMARKS),
    )
    ablate_parser.add_argument("--seeds", type=int, default=5, help="use seeds 1..N")
    ablate_parser.add_argument(
        "--episodes", type=int, default=200, help="evaluation budget per run"
    )
    ablate_parser.set_defaults(func=command_ablate)

    report_parser = subparsers.add_parser("report", help="generate the ablation report")
    report_parser.add_argument("--experiment", required=True)
    report_parser.add_argument(
        "--noise-experiment",
        default=None,
        help="a larger-seed experiment used to measure the seed noise floor",
    )
    report_parser.add_argument("--out", default=None, help="output directory")
    report_parser.set_defaults(func=command_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
