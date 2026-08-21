"""Behaviour through training, for the DRC and the two architectures swapped in for it.

    uv run python scripts/arch_swap_report.py RUNS_DIR [--out figures] [--min-steps 40000000]

For each architecture, the proxy run (rho=1.0) and its control (rho=0.5):
cleanba's in-training evaluation returns at rho 1.0 / 0.5 / 0.0, and the
rho=1.0 - rho=0.0 gap, against training steps. That is Experiment 1 figure 6
drawn three times, one architecture per column, so "the proxy is learnt after
competence" can be read off each. The table underneath gives the numbers:
when the return curve reaches 95% of its final level (competence), the settled
gap, and when the gap first reaches half its settled value.

Reads ``metrics.csv`` from each run directory, which is what cleanba's CSV
writer leaves beside the checkpoints. A missing run is reported and skipped,
so this can be run while training is still in progress.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK2, MUTED = "#52514e", "#9a9992"

ARCHITECTURES = {
    "DRC(3,3)": ("maze11.s1234", "clean11fv.s1234"),
    "ResNet": ("resnet11.s1234", "resnet11clean.s1234"),
    "Transformer": ("vit11.s1234", "vit11clean.s1234"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, help="Directory holding the run directories.")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--name", type=str, default="fig_arch_swap_dynamics")
    parser.add_argument(
        "--settle-fraction",
        type=float,
        default=0.5,
        help="The gap is 'settled' over the last this-fraction of training; 0.5 mirrors Experiment 1's 'after 40M of 150M' closely enough.",
    )
    return parser.parse_args()


def arm(frame: pd.DataFrame, name: str) -> pd.Series:
    column = f"{name}/00_episode_returns"
    return frame[column].dropna().sort_index() if column in frame else pd.Series(dtype=float)


def load(path: Path) -> pd.DataFrame | None:
    csv = path / "metrics.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv, index_col=0).sort_index()


def gap_of(frame: pd.DataFrame) -> pd.Series:
    hi, lo = arm(frame, "rho100"), arm(frame, "rho000")
    idx = hi.index.intersection(lo.index)
    return hi.loc[idx] - lo.loc[idx]


def competence_step(returns: pd.Series, fraction: float = 0.95) -> float:
    """First evaluation at which the return reaches ``fraction`` of the way from its start to its end."""
    if len(returns) < 3:
        return float("nan")
    start, end = returns.iloc[0], returns.iloc[-len(returns) // 4 :].mean()
    target = start + fraction * (end - start)
    reached = returns[returns >= target]
    return float(reached.index[0]) if len(reached) else float("nan")


def sustained_from(series: pd.Series, level: float) -> float:
    """First step after which the series never again drops below ``level``.

    A single noisy early evaluation can touch the level long before anything
    has changed; what the question needs is when the gap *arrives*, so the
    last dip below the level is what dates it.
    """
    below = series[series < level]
    if not len(below):
        return float(series.index[0]) if len(series) else float("nan")
    later = series[series.index > below.index[-1]]
    return float(later.index[0]) if len(later) else float("nan")


def main() -> None:
    args = parse_args()
    fig, axes = plt.subplots(2, len(ARCHITECTURES), figsize=(4.2 * len(ARCHITECTURES), 5.4), sharex="col")
    rows = []
    for column, (label, (proxy_name, control_name)) in enumerate(ARCHITECTURES.items()):
        ax_top, ax_bottom = axes[0, column], axes[1, column]
        proxy, control = load(args.runs / proxy_name), load(args.runs / control_name)
        ax_top.set_title(label)
        if proxy is None:
            ax_top.text(0.5, 0.5, f"{proxy_name}\nnot found", ha="center", va="center", transform=ax_top.transAxes)
        else:
            for name, colour, legend in (
                ("rho100", BLUE, "ρ = 1.0"),
                ("rho050", AQUA, "ρ = 0.5"),
                ("rho000", ORANGE, "ρ = 0.0"),
            ):
                s = arm(proxy, name)
                ax_top.plot(s.index / 1e6, s.values, "-", color=colour, lw=1.8, label=legend)
        ax_top.grid(axis="y", alpha=0.4)
        if column == 0:
            ax_top.set_ylabel("evaluation return (proxy run)")
            ax_top.legend(loc="lower right", fontsize=8, ncol=3)

        for frame, run_name, colour, legend in (
            (proxy, proxy_name, BLUE, "trained at ρ = 1.0 (proxy available)"),
            (control, control_name, ORANGE, "trained at ρ = 0.5 (control)"),
        ):
            if frame is None:
                rows.append((label, run_name, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            gap = gap_of(frame)
            ax_bottom.plot(gap.index / 1e6, gap.values, "-o", color=colour, lw=1.8, ms=3, label=legend)
            total = float(gap.index.max()) if len(gap) else float("nan")
            settled = gap[gap.index >= (1 - args.settle_fraction) * total].mean() if len(gap) else float("nan")
            half_step = sustained_from(gap, settled / 2) if settled > 0 else float("nan")
            returns = arm(frame, "rho100")
            rows.append(
                (
                    label,
                    run_name,
                    total,
                    competence_step(returns),
                    float(returns.iloc[-max(1, len(returns) // 4) :].mean()) if len(returns) else np.nan,
                    settled,
                    half_step,
                )
            )
        ax_bottom.axhline(0, color=MUTED, lw=1)
        ax_bottom.grid(axis="y", alpha=0.4)
        ax_bottom.set_xlabel("training steps (millions)")
        if column == 0:
            ax_bottom.set_ylabel("gap  (ρ=1.0 − ρ=0.0)")
            ax_bottom.legend(loc="lower right", fontsize=8)

    # One vertical scale for every gap panel, so the columns compare by eye.
    lows = [ax.get_ylim()[0] for ax in axes[1]]
    highs = [ax.get_ylim()[1] for ax in axes[1]]
    for ax in axes[1]:
        ax.set_ylim(min(lows + [-0.05]), max(highs + [0.05]))

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{args.name}.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {args.out / args.name}.png / .pdf\n")

    header = f"{'architecture':>13}{'run':>22}{'steps':>8}{'competent@':>12}{'return':>9}{'gap':>9}{'gap half@':>11}"
    print(header)
    for label, run_name, total, comp, ret, settled, half_step in rows:
        fmt = lambda v, scale=1e6, digits=1: "-" if v != v else f"{v / scale:.{digits}f}M"  # noqa: E731
        print(
            f"{label:>13}{run_name:>22}{fmt(total, digits=0):>8}{fmt(comp):>12}"
            f"{'-' if ret != ret else f'{ret:+.3f}':>9}{'-' if settled != settled else f'{settled:+.3f}':>9}{fmt(half_step):>11}"
        )
    print(
        "\ncompetent@ = first evaluation where the rho=1.0 return is 95% of the way to its final level;"
        "\ngap = mean rho1.0-rho0.0 return gap over the settled part of training; gap half@ = first"
        "\nevaluation after which the gap stays above half that. 'Proxy after competence' is gap half@ > competent@."
    )


if __name__ == "__main__":
    main()
