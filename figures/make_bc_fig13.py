"""Figure 13's threshold-against-value plot, for the imitation-trained route model.

Same layout and reading as ``make_wide_figures.py`` on the DRC: one row per
seed, one panel per swept colour, the exchange rate (extra steps walked for
colour 0) against what the swept colour is worth. Blue is a fine-tuned arm,
orange is the axis fitted without that arm and written into the base, the
dashed line is the expert, the dotted line is the untouched base.

Two differences from the DRC figure, both forced by what was run:

* There are no arms trained past the flip (the ``x`` sweeps), so the
  past-the-flip regime is shown only by the axis written beyond the grid
  (open squares), at offsets ±0.6 and ±0.9.
* The deepening writes beyond the grid over-carry far enough (up to ~140
  steps at +0.9) that plotting them would flatten everything else, so the
  y-axis is clipped and an off-scale point is drawn at the edge with its
  value written beside it.

The expert here is the demonstrator's own rule, ``value - 0.05 x distance``
with no discount, so its threshold is ``20 (v_0 - v_1)`` steps: a straight
line, where the DRC's discounted optimum is slightly curved.

Reads ``figures/data/bc/value_axis.<base>.<sweep>.json`` (written by 027) and
writes ``figures/fig_bc_wide_threshold.{png,pdf}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).parent / "data" / "bc"
OUT = Path(__file__).parent

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9992"
SURFACE = "#fcfcfb"

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK,
        "legend.frameon": False,
    }
)

BASE_VALUES = (1.0, 0.5)
STEP_PENALTY = 0.05
Y_LIM = (-12.0, 42.0)


def expert_threshold(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Extra steps the demonstrator walks for colour 0: ``(v0 - v1) / penalty``."""
    return (v0 - v1) / STEP_PENALTY


def swept_value(sweep: str, offset: float) -> float:
    return BASE_VALUES[int(sweep[1])] + offset


def load(base: str, sweep: str) -> dict:
    return json.loads((DATA / f"value_axis.{base}.{sweep}.json").read_text())


def series(ax, xs, ys, colour, label, marker, **kw):
    order = np.argsort(xs)
    ax.plot(
        np.asarray(xs)[order],
        np.asarray(ys)[order],
        marker,
        color=colour,
        ms=4.2,
        lw=1.6,
        ls="-",
        mec=SURFACE,
        mew=0.7,
        zorder=3,
        label=label,
        **kw,
    )


def beyond_grid(ax, xs, ys, colour, label):
    """Open squares for writes outside the fitted grid; off-scale ones pinned to the edge."""
    first = True
    for x, y in sorted(zip(xs, ys)):
        shown = min(max(y, Y_LIM[0] + 0.8), Y_LIM[1] - 0.8)
        ax.plot(
            [x],
            [shown],
            "s",
            color=colour,
            mfc=SURFACE,
            ms=5.0,
            mew=1.4,
            zorder=4,
            label=label if first else None,
        )
        first = False
        if shown != y:
            ax.annotate(
                f"{y:.0f}" + (" ↑" if y > shown else " ↓"),
                xy=(x, shown),
                xytext=(0, 7 if y > shown else -12),
                textcoords="offset points",
                ha="center",
                color=colour,
                fontsize=7.5,
            )


def main() -> None:
    bases = sorted({p.name[len("value_axis.") : -len(".o0.json")] for p in DATA.glob("value_axis.*.json")})
    fig, axes = plt.subplots(len(bases), 2, figsize=(10.5, 3.4 * len(bases)), squeeze=False)

    for row, base in enumerate(bases):
        for col, sweep in enumerate(("o0", "o1")):
            ax = axes[row][col]
            d = load(base, sweep)
            swept = int(sweep[1])
            other = BASE_VALUES[1 - swept]
            arms = d["behaviour"]["arms"]
            offsets = [float(o) for o in arms]
            values = [swept_value(sweep, o) for o in offsets]

            # The expert, over the span the panel shows (grid plus the writes beyond it).
            lo, hi = min(values) - 0.5, max(values) + 0.5
            grid = np.linspace(lo, hi, 200)
            v0, v1 = (grid, np.full_like(grid, other)) if swept == 0 else (np.full_like(grid, other), grid)
            opt = expert_threshold(v0, v1)
            ax.plot(grid, opt, color=MUTED, lw=1.1, ls="--", zorder=1)
            peak = int(np.argmax(opt)) if swept == 0 else int(np.argmin(grid))
            ax.annotate(
                "expert",
                xy=(grid[peak], opt[peak]),
                xytext=(8, -10),
                textcoords="offset points",
                color=MUTED,
                fontsize=8,
            )
            ax.axhline(0, color=INK2, lw=0.8, alpha=0.45, zorder=1)
            ax.axhline(d["behaviour"]["base"]["indifference"], color=MUTED, lw=0.7, ls=":", zorder=0)

            series(ax, values, [arms[o]["arm"]["indifference"] for o in arms], BLUE, "trained: one fine-tune per value", "o")
            series(
                ax,
                values,
                [arms[o]["written"]["indifference"] for o in arms],
                ORANGE,
                "the axis, written (that value held out)",
                "s",
            )

            controls = d["behaviour"]["controls"]
            extra = {k: v for k, v in controls.items() if k.startswith("extrapolate_")}
            if extra:
                xs = [swept_value(sweep, float(k.split("_")[1])) for k in extra]
                ys = [v["indifference"] for v in extra.values()]
                beyond_grid(ax, xs, ys, ORANGE, "the same axis, written beyond the grid")

            ax.set_ylim(*Y_LIM)
            ax.set_xlim(lo, hi)
            ax.set_title(
                f"colour {swept} swept, colour {1 - swept} fixed at {other:g}\n{base}  ·  {len(arms)} arms",
                color=INK,
            )
            ax.set_xlabel(f"what colour {swept} is worth")
            ax.grid(axis="y", color=MUTED, alpha=0.22, lw=0.6)
        axes[row][0].set_ylabel("extra steps walked for colour 0")

    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][0].legend(handles, labels, loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Route model (imitation, values hidden): the exchange rate each arm learnt, and the one the axis writes",
        color=INK,
        fontsize=10.5,
        y=0.995,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_bc_wide_threshold.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / 'fig_bc_wide_threshold.png'}")


if __name__ == "__main__":
    main()
