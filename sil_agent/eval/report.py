"""The ablation report — tables and plots, generated entirely from the database.

Nothing here reads a log file. The episodes table is the durable record, so an
analysis built on it can be regenerated months later from the database alone,
long after the console output that produced it has scrolled away. That is also
why the report is never hand-edited: if a number looks wrong, the fix goes in
the code and the report is regenerated.

The output is a directory containing ``report.md`` and one convergence plot per
benchmark.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported. "Agg" is a file-only backend with no
# GUI: without it matplotlib tries to find a display, which hangs or warns under
# pytest and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import mannwhitneyu

from sil_agent.eval.metrics import (
    CellStats,
    NoiseFloor,
    RunSummary,
    seed_noise_floor,
    summarise_cell,
    summarise_run,
)
from sil_agent.persistence.repo import RunRepository
from sil_agent.simulators.toy import BENCHMARKS

# ---------------------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------------------
#
# Colours are the first three slots of a categorical palette validated for
# colour-vision deficiency: worst all-pairs CVD separation ΔE 9.2, worst
# normal-vision separation ΔE 24.0. Assigned to strategies in a fixed order and
# never cycled, so a strategy keeps its colour across every plot in the report —
# a reader who learns "orange is grid search" on one chart is not re-taught on
# the next.
SERIES_COLOURS: list[str] = ["#2a78d6", "#eb6834", "#1baf7a"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Regret is clamped at zero, and a log axis cannot show zero. Values below this
# are drawn at this floor; the plot caption says so.
LOG_FLOOR = 1e-6


@dataclass(frozen=True)
class Report:
    """Where the generated artefacts ended up."""

    directory: Path
    markdown: Path
    plots: tuple[Path, ...]


def colour_for(strategy: str, order: Sequence[str]) -> str:
    """A strategy's fixed colour slot.

    Beyond the third strategy the palette stops being CVD-safe for all pairs.
    Phase 4 will have six strategies, at which point this needs facetting into
    small multiples rather than a fourth hue — flagged here rather than
    discovered then.
    """
    index = list(order).index(strategy)
    return SERIES_COLOURS[index % len(SERIES_COLOURS)]


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------


def gather(experiment: str, repo: RunRepository) -> list[RunSummary]:
    """Load and score every run in an experiment."""
    summaries: list[RunSummary] = []

    for stored in repo.list_runs(experiment):
        benchmark = BENCHMARKS.get(stored.simulator)
        if benchmark is None:
            # A run against a simulator this build does not know about. Skipping
            # is right: scoring it would need a `known_optimum` that is not
            # available, and inventing one would corrupt every regret in the
            # table.
            continue
        summaries.append(summarise_run(stored, benchmark))

    return summaries


def cells_by_strategy_and_simulator(
    summaries: Sequence[RunSummary],
) -> dict[tuple[str, str], CellStats]:
    grouped: dict[tuple[str, str], list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.strategy, summary.simulator)].append(summary)
    return {key: summarise_cell(runs) for key, runs in grouped.items()}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def convergence_plot(
    summaries: Sequence[RunSummary],
    simulator: str,
    strategies: Sequence[str],
    path: Path,
) -> Path:
    """Best-so-far regret against evaluation count, median with interquartile band.

    Median and IQR rather than mean and standard deviation. Regret distributions
    are skewed — one unlucky seed on Rosenbrock is orders of magnitude worse than
    the rest — and a mean would be dragged around by it while the band went
    negative and vanished off a log axis.
    """
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    # Collected while plotting, placed afterwards: where a label belongs cannot
    # be decided until every line is drawn and the axis limits are known.
    endpoints: list[tuple[float, str, str]] = []

    for strategy in strategies:
        runs = [s for s in summaries if s.strategy == strategy and s.simulator == simulator]
        if not runs:
            continue

        colour = colour_for(strategy, strategies)
        length = max(len(r.regret_curve) for r in runs)
        x_values = list(range(1, length + 1))

        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []

        for index in range(length):
            at_index = [
                r.regret_curve[index]
                for r in runs
                if index < len(r.regret_curve) and r.regret_curve[index] is not None
            ]
            values = sorted(v for v in at_index if v is not None)
            if not values:
                # No seed has found a feasible point yet at this evaluation.
                medians.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
                continue
            medians.append(max(_median(values), LOG_FLOOR))
            lows.append(max(_percentile(values, 25.0), LOG_FLOOR))
            highs.append(max(_percentile(values, 75.0), LOG_FLOOR))

        axes.fill_between(x_values, lows, highs, color=colour, alpha=0.15, linewidth=0)
        axes.plot(x_values, medians, color=colour, linewidth=2.0, label=strategy)

        last = _last_finite(medians)
        if last is not None:
            endpoints.append((last[1], strategy, colour))

    axes.set_yscale("log")
    axes.set_xlabel("simulator evaluations", color=INK_MUTED, fontsize=9)
    axes.set_ylabel("regret (log scale)", color=INK_MUTED, fontsize=9)
    axes.set_title(
        f"{simulator} — best-so-far regret, median with interquartile band",
        color=INK_PRIMARY,
        fontsize=10,
        loc="left",
    )

    axes.grid(True, color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=INK_MUTED, labelsize=8)

    axes.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY, loc="upper right")

    _place_end_labels(axes, endpoints)

    # Reserve the right-hand fifth of the figure for those labels. Without this
    # `tight_layout` fits the axes to the full width and the labels are clipped
    # off the edge of the image.
    figure.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)

    return path


def _place_end_labels(axes: Axes, endpoints: Sequence[tuple[float, str, str]]) -> None:
    """Label each line at its right-hand end, nudged apart so labels never overlap.

    Two strategies that converge to nearly the same regret — which is exactly
    what happens when neither of them works — end up with their labels on top of
    each other, and the chart becomes unreadable at precisely the point it has
    something to say. So the labels are laid out in axis-fraction space with a
    minimum separation enforced, which decouples them from the data values
    without moving the lines.
    """
    if not endpoints:
        return

    bottom, top = axes.get_ylim()
    span = math.log10(top) - math.log10(bottom)
    if span <= 0:
        return

    def to_fraction(value: float) -> float:
        return (math.log10(value) - math.log10(bottom)) / span

    ordered = sorted(endpoints, key=lambda item: item[0])
    fractions = [to_fraction(value) for value, _, _ in ordered]

    # One upward pass: push any label that is too close to the one below it.
    minimum_gap = 0.055
    for index in range(1, len(fractions)):
        if fractions[index] - fractions[index - 1] < minimum_gap:
            fractions[index] = fractions[index - 1] + minimum_gap

    # If that pushed the top label off the axes, slide the whole stack down.
    overflow = fractions[-1] - 1.0
    if overflow > 0:
        fractions = [fraction - overflow for fraction in fractions]

    for fraction, (_, strategy, colour) in zip(fractions, ordered, strict=True):
        axes.annotate(
            strategy,
            xy=(1.015, min(max(fraction, 0.0), 1.0)),
            xycoords="axes fraction",
            color=colour,
            fontsize=8,
            va="center",
            annotation_clip=False,
        )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(values: Sequence[float], pct: float) -> float:
    from sil_agent.eval.metrics import percentile

    return percentile(values, pct)


def _last_finite(values: Sequence[float]) -> tuple[int, float] | None:
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value == value:  # not NaN
            return index, value
    return None


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def headline_table(
    cells: dict[tuple[str, str], CellStats],
    strategies: Sequence[str],
    simulators: Sequence[str],
) -> str:
    lines = ["| Strategy | " + " | ".join(simulators) + " |"]
    lines.append("|---" * (len(simulators) + 1) + "|")

    for strategy in strategies:
        row = [strategy]
        for simulator in simulators:
            cell = cells.get((strategy, simulator))
            row.append(cell.label if cell else "—")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def budget_table(
    cells: dict[tuple[str, str], CellStats],
    strategies: Sequence[str],
    simulators: Sequence[str],
) -> str:
    """How much of the budget each strategy actually used, and why it stopped.

    Without this the headline table is misleading: a strategy that stopped at 64
    of 200 evaluations because it ran out of grid did not lose the same contest
    the others were in, and the report has to say so.
    """
    lines = ["| Strategy | Benchmark | Mean evaluations | Terminated |"]
    lines.append("|---|---|---|---|")

    for strategy in strategies:
        for simulator in simulators:
            cell = cells.get((strategy, simulator))
            if cell is None:
                continue
            lines.append(
                f"| {strategy} | {simulator} | {cell.mean_evaluations:.0f} | "
                f"{', '.join(cell.terminations)} |"
            )

    return "\n".join(lines)


def comparison_table(
    cells: dict[tuple[str, str], CellStats],
    strategies: Sequence[str],
    simulators: Sequence[str],
) -> str:
    """Pairwise Mann-Whitney U between strategies, per benchmark.

    A *rank* test: it asks whether one strategy's results tend to sit below the
    other's, without assuming the values are normally distributed — which regret
    values are not. A t-test on five skewed samples would put a precise-looking
    number on an assumption that does not hold.

    With five seeds per group the smallest achievable two-sided p is 0.0079, so
    the test can only detect large effects. That is a real limit of the protocol
    and is stated rather than worked around.
    """
    header = [
        "| Benchmark | A | B | median A | median B | U | p (two-sided) | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines = list(header)
    any_pseudoreplicated = False

    for simulator in simulators:
        for i, first in enumerate(strategies):
            for second in strategies[i + 1 :]:
                left = cells.get((first, simulator))
                right = cells.get((second, simulator))
                if left is None or right is None:
                    continue
                if len(left.values) < 3 or len(right.values) < 3:
                    continue

                statistic, p_value = mannwhitneyu(
                    left.values, right.values, alternative="two-sided"
                )

                # A deterministic strategy contributes the same number once per
                # seed. Those are not five observations, they are one observation
                # written down five times, and a rank test given them reports a
                # confidence it has not earned.
                flag = ""
                if _has_no_variance(left) or _has_no_variance(right):
                    flag = "†"
                    any_pseudoreplicated = True

                lines.append(
                    f"| {simulator} | {first} | {second} | "
                    f"{left.median:.4g} | {right.median:.4g} | "
                    f"{statistic:.0f} | {p_value:.4f} | {flag} |"
                )

    if any_pseudoreplicated:
        lines += [
            "",
            "† One side of this comparison is deterministic: every seed produced an",
            "identical run. Those are not independent observations, they are one",
            "observation recorded once per seed, so the p-value overstates the evidence —",
            "the test cannot tell replication from repetition. Read these rows as a",
            "comparison of medians and disregard the p-value.",
        ]

    if len(lines) == len(header):
        return (
            "_Not enough seeds per cell to run a rank test — at least 3 are needed, "
            "and the protocol's 5 is the practical minimum for a two-sided result._"
        )

    return "\n".join(lines)


def _has_no_variance(cell: CellStats) -> bool:
    """True when every seed in this cell produced the same number.

    The signature of a deterministic strategy. Worth detecting rather than
    assuming, because it is also what a broken harness looks like — one that
    passed the same seed to every run would produce exactly this.
    """
    return len(cell.values) > 1 and len(set(cell.values)) == 1


def noise_table(floors: Sequence[NoiseFloor]) -> str:
    lines = [
        "| Strategy | Benchmark | Seeds run | Mean regret "
        "| 90% interval of a 5-seed mean | Spread |",
        "|---|---|---|---|---|---|",
    ]
    for floor in floors:
        lines.append(
            f"| {floor.strategy} | {floor.simulator} | {floor.population} | "
            f"{floor.observed_mean:.4g} | [{floor.low:.4g}, {floor.high:.4g}] | "
            f"{floor.spread:.4g} |"
        )
    return "\n".join(lines)


def per_seed_table(summaries: Sequence[RunSummary]) -> str:
    """Every raw number. The answer to "are you sure you didn't pick a good seed?"."""
    lines = ["| Strategy | Benchmark | Seed | Evaluations | Best feasible | Regret | Terminated |"]
    lines.append("|---|---|---|---|---|---|---|")

    for summary in sorted(
        summaries, key=lambda s: (s.strategy, s.simulator, s.seed)
    ):
        best = f"{summary.best_objective:.6g}" if summary.best_objective is not None else "none"
        regret = f"{summary.regret:.6g}" if summary.regret is not None else "—"
        termination = summary.termination.value if summary.termination else "INCOMPLETE"
        lines.append(
            f"| {summary.strategy} | {summary.simulator} | {summary.seed} | "
            f"{summary.evaluations} | {best} | {regret} | {termination} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build(
    experiment: str,
    repo: RunRepository,
    out_dir: Path,
    *,
    noise_experiment: str | None = None,
    subsample: int = 5,
) -> Report:
    """Generate the full report for an experiment.

    ``noise_experiment`` names a second, larger experiment — the same strategies
    over many more seeds — used only to measure how much a five-seed mean moves
    by chance. It is optional; without it that section is omitted rather than
    guessed at.
    """
    summaries = gather(experiment, repo)
    if not summaries:
        raise ValueError(f"no runs found for experiment {experiment!r}")

    strategies = sorted({s.strategy for s in summaries})
    simulators = sorted({s.simulator for s in summaries})
    cells = cells_by_strategy_and_simulator(summaries)

    out_dir.mkdir(parents=True, exist_ok=True)

    plots: list[Path] = []
    for simulator in simulators:
        path = out_dir / f"convergence_{simulator}.png"
        plots.append(convergence_plot(summaries, simulator, strategies, path))

    floors: list[NoiseFloor] = []
    if noise_experiment:
        noise_summaries = gather(noise_experiment, repo)
        noise_cells = cells_by_strategy_and_simulator(noise_summaries)
        for (strategy, simulator), cell in sorted(noise_cells.items()):
            floor = seed_noise_floor(
                cell.values,
                strategy=strategy,
                simulator=simulator,
                subsample=subsample,
            )
            if floor is not None:
                floors.append(floor)

    incomplete = [s for s in summaries if not s.is_complete]
    max_evaluations = max(s.max_evaluations for s in summaries)

    sections: list[str] = [
        f"# Ablation report — `{experiment}`",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC, "
        f"from {len(summaries)} runs in the `episodes` table.",
        "",
        f"- Strategies: {', '.join(strategies)}",
        f"- Benchmarks: {', '.join(simulators)}",
        f"- Seeds per cell: {len({s.seed for s in summaries})}",
        f"- Evaluation budget: {max_evaluations} simulator calls per run",
        "",
        "Every strategy receives the same benchmarks, the same seeds and the same",
        "number of simulator calls. Scores are **regret** — distance from the",
        "benchmark's known global optimum — computed over feasible results only.",
        "",
        "## Headline — mean ± standard deviation of final regret",
        "",
        "Lower is better.",
        "",
        headline_table(cells, strategies, simulators),
        "",
        "Grid search shows a standard deviation of exactly zero because it does not",
        "consume the seed: its points are fixed by the parameter space and the budget,",
        "so every seed produces the identical run. That is a property of the strategy,",
        "not a suspiciously clean measurement.",
        "",
        "## Budget actually used, and why each run stopped",
        "",
        budget_table(cells, strategies, simulators),
        "",
        "`EXHAUSTED` means the strategy ran out of things to propose before it ran",
        "out of budget — grid search covers its whole grid and stops. `BUDGET`",
        "means the full allowance of simulator calls was spent.",
        "",
        "## Is the difference real?",
        "",
        comparison_table(cells, strategies, simulators),
        "",
    ]

    if floors:
        sections += [
            "## Seed noise floor",
            "",
            "Measured by running each baseline over many more seeds than the protocol",
            f"requires, then repeatedly drawing {subsample} of them at random and taking the",
            "mean. The spread column is how far apart two "
            f"{subsample}-seed means can be when",
            "nothing differs but the seeds — **a gap smaller than that proves nothing.**",
            "",
            noise_table(floors),
            "",
        ]

    sections += ["## Convergence", ""]
    for path, simulator in zip(plots, simulators, strict=True):
        sections += [
            f"### {simulator}",
            "",
            f"![convergence on {simulator}]({path.name})",
            "",
        ]
    sections += [
        f"Regret is clamped at zero and plotted on a log axis; values below {LOG_FLOOR:g}",
        "are drawn at that floor.",
        "",
    ]

    if incomplete:
        sections += [
            "## Incomplete runs",
            "",
            f"{len(incomplete)} run(s) had not finished when this report was generated.",
            "Re-run the harness with the same experiment name to continue them.",
            "",
        ]

    sections += [
        "## Every run",
        "",
        "The raw numbers behind every cell above, so the table can be checked",
        "rather than trusted.",
        "",
        per_seed_table(summaries),
        "",
    ]

    markdown_path = out_dir / "report.md"
    markdown_path.write_text("\n".join(sections), encoding="utf-8")

    return Report(directory=out_dir, markdown=markdown_path, plots=tuple(plots))
