"""Regret against evaluation budget — where does a cheap strategy stop winning?

The Phase 3.5 headline is that the agent beats TPE at 20 evaluations. On its own
that sentence invites a reading it does not support, because TPE at 200
evaluations reaches regret an order of magnitude below anything measured here.
The honest claim is about **sample efficiency under a tight budget**, and a claim
about budgets has to be plotted against budget.

Each point is an **independent run at that budget**, never a prefix of a longer
one. That distinction matters for the agent and not for TPE: the planner is told
`max_evaluations` in its prompt, so a run that knows it has eighty evaluations
explores differently from one that knows it has ten. Truncating the long run
would quietly measure the wrong thing.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from sil_agent.eval.metrics import CellStats, RunSummary, summarise_cell
from sil_agent.eval.report import (
    AXIS,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    LOG_FLOOR,
    SURFACE,
    colour_for,
    gather,
    linestyle_for,
)
from sil_agent.persistence.repo import RunRepository


@dataclass(frozen=True)
class SweepPoint:
    """One (strategy, simulator, budget) cell."""

    strategy: str
    simulator: str
    budget: int
    stats: CellStats

    @property
    def median(self) -> float | None:
        return self.stats.median


def gather_sweep(experiments: Sequence[str], repo: RunRepository) -> list[SweepPoint]:
    """Collect every experiment into (strategy, simulator, budget) cells.

    The budget is read from the runs rather than supplied, so an experiment
    cannot be filed under a budget it was not actually run at.
    """
    grouped: dict[tuple[str, str, int], list[RunSummary]] = {}

    for experiment in experiments:
        for summary in gather(experiment, repo):
            if not summary.is_complete:
                continue
            key = (summary.strategy, summary.simulator, summary.max_evaluations)
            grouped.setdefault(key, []).append(summary)

    points = [
        SweepPoint(strategy=strategy, simulator=simulator, budget=budget,
                   stats=summarise_cell(runs))
        for (strategy, simulator, budget), runs in grouped.items()
    ]
    points.sort(key=lambda p: (p.simulator, p.strategy, p.budget))
    return points


def curve_for(
    points: Sequence[SweepPoint], strategy: str, simulator: str
) -> list[tuple[int, float]]:
    """Budget -> median regret, ascending, skipping cells with no feasible result."""
    return [
        (point.budget, point.median)
        for point in sorted(points, key=lambda p: p.budget)
        if point.strategy == strategy
        and point.simulator == simulator
        and point.median is not None
    ]


@dataclass(frozen=True)
class Crossover:
    """Where two curves swap places, as an interval and an interpolated estimate."""

    simulator: str
    winner_below: str
    winner_above: str
    lower_budget: int
    upper_budget: int
    estimate: float

    def describe(self) -> str:
        return (
            f"{self.simulator}: {self.winner_below} leads up to {self.lower_budget} "
            f"evaluations, {self.winner_above} from {self.upper_budget} on; "
            f"crossing near **{self.estimate:.0f}**"
        )


def find_crossover(
    points: Sequence[SweepPoint], first: str, second: str, simulator: str
) -> Crossover | None:
    """The budget at which the leader changes, or None if it never does.

    Interpolated in log-log space, because both regret and budget span orders of
    magnitude and a straight line between two points on linear axes would put the
    crossing in the wrong place. With four budgets this is an estimate bracketed
    by two measured points, and `describe` reports the bracket as well as the
    number — a precise-looking crossover from four points would be false
    precision.
    """
    left = dict(curve_for(points, first, simulator))
    right = dict(curve_for(points, second, simulator))
    budgets = sorted(set(left) & set(right))
    if len(budgets) < 2:
        return None

    def gap(budget: int) -> float:
        """Positive when `first` is worse (higher regret) than `second`."""
        return math.log(max(left[budget], LOG_FLOOR)) - math.log(
            max(right[budget], LOG_FLOOR)
        )

    for lower, upper in itertools.pairwise(budgets):
        low_gap, high_gap = gap(lower), gap(upper)
        if low_gap == 0 or (low_gap < 0) == (high_gap < 0):
            continue  # same leader at both ends; no crossing in this interval

        # Linear interpolation of the gap in log-budget.
        span = math.log(upper) - math.log(lower)
        fraction = low_gap / (low_gap - high_gap)
        estimate = math.exp(math.log(lower) + fraction * span)

        below, above = (second, first) if low_gap > 0 else (first, second)
        return Crossover(
            simulator=simulator,
            winner_below=below,
            winner_above=above,
            lower_budget=lower,
            upper_budget=upper,
            estimate=estimate,
        )

    return None


def swept_strategies(points: Sequence[SweepPoint], simulator: str) -> list[str]:
    """Strategies actually measured at more than one budget.

    A strategy present at a single budget contributes one number and a row of
    dashes, which reads as missing data rather than as "not part of this
    experiment". The sweep is about slopes, so a single point is not on it.
    """
    budgets_by_strategy: dict[str, set[int]] = {}
    for point in points:
        if point.simulator == simulator:
            budgets_by_strategy.setdefault(point.strategy, set()).add(point.budget)
    return sorted(name for name, budgets in budgets_by_strategy.items() if len(budgets) > 1)


def expected_n(points: Sequence[SweepPoint], simulator: str) -> int:
    """The seed count most cells on this sweep were built from."""
    counts = [
        p.stats.n for p in points
        if p.simulator == simulator and p.strategy in swept_strategies(points, simulator)
    ]
    return max(counts, default=0)


def uneven_cells(points: Sequence[SweepPoint], simulator: str) -> list[SweepPoint]:
    """Cells built from fewer seeds than the rest of the sweep.

    A run that was interrupted and never resumed leaves a cell with, say, two
    seeds where its neighbours have five. The median still prints, the curve
    still draws, and nothing on the chart says the point is weaker — so this
    finds them and the report names them.
    """
    target = expected_n(points, simulator)
    swept = swept_strategies(points, simulator)
    return [
        p for p in points
        if p.simulator == simulator and p.strategy in swept and 0 < p.stats.n < target
    ]


def significance_row(
    points: Sequence[SweepPoint], simulator: str, first: str, second: str
) -> list[tuple[int, int, float, str]]:
    """Per-budget rank test between two strategies: (budget, n, p, leader).

    Without this the sweep is a pair of medians and an invitation to read a
    crossing as a result. On this data the medians swap places twice and only
    one budget carries a difference the seeds actually support — which is the
    opposite of what the curves suggest at a glance.
    """
    budgets = sorted({p.budget for p in points if p.simulator == simulator})
    lookup = {
        (p.strategy, p.budget): p for p in points if p.simulator == simulator
    }

    out: list[tuple[int, int, float, str]] = []
    for budget in budgets:
        left = lookup.get((first, budget))
        right = lookup.get((second, budget))
        if left is None or right is None:
            continue
        if len(left.stats.values) < 2 or len(right.stats.values) < 2:
            continue
        _, p_value = mannwhitneyu(
            left.stats.values, right.stats.values, alternative="two-sided", method="exact"
        )
        leader = first if (left.median or 0) < (right.median or 0) else second
        out.append((budget, min(left.stats.n, right.stats.n), float(p_value), leader))
    return out


def sweep_table(points: Sequence[SweepPoint], simulator: str) -> str:
    """Median final regret at each budget, one row per swept strategy."""
    strategies = swept_strategies(points, simulator)
    budgets = sorted(
        {p.budget for p in points if p.simulator == simulator and p.strategy in strategies}
    )

    lookup: Mapping[tuple[str, int], SweepPoint] = {
        (p.strategy, p.budget): p for p in points if p.simulator == simulator
    }

    header = "| Strategy | " + " | ".join(f"{b} evals" for b in budgets) + " |"
    divider = "|---" * (len(budgets) + 1) + "|"
    lines = [header, divider]

    for strategy in strategies:
        cells = []
        for budget in budgets:
            point = lookup.get((strategy, budget))
            if point is None or point.median is None:
                cells.append("—")
            else:
                cells.append(f"{point.median:.4g}")
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def sweep_plot(
    points: Sequence[SweepPoint], simulator: str, strategies: Sequence[str], path: Path
) -> Path:
    """Median final regret against budget, log-log."""
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    for strategy in strategies:
        curve = curve_for(points, strategy, simulator)
        if not curve:
            continue
        budgets = [b for b, _ in curve]
        regrets = [max(r, LOG_FLOOR) for _, r in curve]
        axes.plot(
            budgets,
            regrets,
            color=colour_for(strategy, strategies),
            linestyle=linestyle_for(strategy, strategies),
            linewidth=2.0,
            marker="o",
            markersize=6,
            label=strategy,
        )

    axes.set_xscale("log")
    axes.set_yscale("log")

    # Label the budgets that were actually run, as plain integers. Matplotlib's
    # log formatter renders them as "2 x 10^1", which is correct notation and
    # useless here — the reader wants to see "20 evaluations", a number they
    # chose, not its scientific form.
    measured = sorted({b for b, _ in
                       (pair for strategy in strategies
                        for pair in curve_for(points, strategy, simulator))})
    if measured:
        axes.set_xticks(measured)
        axes.set_xticklabels([str(b) for b in measured])
        axes.minorticks_off()

    # The crossover, drawn where it happens. This is the whole point of the
    # chart, and leaving the reader to find it by eye in a log-log plot invites
    # them to find it in the wrong place.
    if len(strategies) == 2:
        crossing = find_crossover(points, strategies[0], strategies[1], simulator)
        if crossing is not None:
            axes.axvline(
                crossing.estimate,
                color=INK_MUTED,
                linewidth=1.0,
                linestyle=(0, (2, 3)),
                zorder=0,
            )
            # "medians cross", not "lead changes". On this data the crossing is
            # not backed by a significant difference at any larger budget, and a
            # chart annotation is exactly where an unsupported claim would get
            # quoted from.
            axes.annotate(
                f"medians cross\n~{crossing.estimate:.0f} evals",
                xy=(crossing.estimate, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4, -6),
                textcoords="offset points",
                color=INK_MUTED,
                fontsize=8,
                va="top",
            )
    axes.set_xlabel("evaluation budget (independent run at each)", color=INK_MUTED, fontsize=9)
    axes.set_ylabel("median final regret", color=INK_MUTED, fontsize=9)
    axes.set_title(
        f"{simulator} — sample efficiency against budget",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    axes.grid(True, color=GRIDLINE, linewidth=0.8, which="both")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=INK_MUTED, labelsize=8)
    axes.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY, loc="lower left")

    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def build(
    experiments: Sequence[str],
    repo: RunRepository,
    out_dir: Path,
    *,
    compare: tuple[str, str] = ("agent_no_reflection", "optuna_tpe"),
) -> Path:
    """Write the sweep report. Returns the markdown path."""
    points = gather_sweep(experiments, repo)
    if not points:
        raise ValueError(f"no complete runs found in {experiments}")

    simulators = sorted({p.simulator for p in points})
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = [
        "# Budget sweep — regret against evaluation count",
        "",
        f"Experiments: {', '.join(experiments)}",
        "",
        "Every point is an **independent run at that budget**, not a prefix of a",
        "longer run. The planner is told its budget in the prompt, so a run that",
        "knows it has 80 evaluations behaves differently from one that knows it has",
        "10; truncating the long run would measure the wrong thing.",
        "",
        "Values are median final regret over the seeds in each cell.",
        "",
    ]

    for simulator in simulators:
        sections += [f"## {simulator}", "", sweep_table(points, simulator), ""]

        # A cell built from fewer seeds than its neighbours is not comparable to
        # them, and a median silently taken over two runs instead of five looks
        # identical in the table. Say so rather than let the curve imply an
        # evenness it does not have.
        uneven = uneven_cells(points, simulator)
        if uneven:
            detail = "; ".join(
                f"{p.strategy} at {p.budget} evals (n={p.stats.n})" for p in uneven
            )
            sections += [
                f"⚠ **Unequal seeds.** Most cells here use {expected_n(points, simulator)} "
                f"seeds; these do not: {detail}. Their medians are drawn from fewer runs "
                "and are less stable than the rest of the curve.",
                "",
            ]

        # Significance first, deliberately. The medians alone invite reading a
        # crossing as a result, and on this data most of the budgets carry no
        # difference the seeds support.
        tests = significance_row(points, simulator, compare[0], compare[1])
        if tests:
            sections += [
                f"### Is the gap real at each budget? (`{compare[0]}` vs `{compare[1]}`)",
                "",
                "| Budget | Leading median | n per group | p (Mann-Whitney, exact, two-sided) | |",
                "|---|---|---|---|---|",
            ]
            for budget, n, p_value, leader in tests:
                verdict = "**significant**" if p_value < 0.05 else "not significant"
                sections.append(
                    f"| {budget} | {leader} | {n} | {p_value:.4f} | {verdict} |"
                )
            sections += [
                "",
                "A leading median with a large p-value means the two distributions "
                "overlap: the ordering is what these particular seeds produced, not a "
                "property of the strategies.",
                "",
            ]

        crossover = find_crossover(points, compare[0], compare[1], simulator)
        significant_budgets = {b for b, _, p, _ in tests if p < 0.05}

        if crossover is None:
            sections += [
                f"The median of `{compare[0]}` and `{compare[1]}` do not cross within "
                "the budgets measured.",
                "",
            ]
        elif significant_budgets and any(
            b > crossover.estimate for b in significant_budgets
        ):
            sections += [f"**Crossover** — {crossover.describe()}", ""]
        else:
            sections += [
                f"The medians cross near {crossover.estimate:.0f} evaluations "
                f"(`{crossover.winner_below}` below, `{crossover.winner_above}` above) — "
                "**but this is not a supported crossover.** No budget above the crossing "
                "shows a significant difference, so what the curves record is one "
                "strategy's advantage fading into noise rather than the other overtaking "
                "it. Reported as a median crossing only.",
                "",
            ]

        plot_path = out_dir / f"sweep_{simulator}.png"
        sweep_plot(points, simulator, swept_strategies(points, simulator), plot_path)
        sections += [f"![budget sweep on {simulator}]({plot_path.name})", ""]

    markdown = out_dir / "sweep.md"
    markdown.write_text("\n".join(sections), encoding="utf-8")
    return markdown
