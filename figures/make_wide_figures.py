"""The threshold-against-value plot, on the wide grid.

    uv run python figures/make_wide_figures.py

The upgrade to ``fig10`` is not that there are more points. The old grid fitted
seven arms over a gap of 0.1 to 0.7 and then *wrote* the axis out to a gap of
-0.7 to +1.3, so the interesting half of that figure was extrapolation. The wide
grid trains arms across gap 0.05 to 0.95, which means most of what used to be
extrapolated is now measured, and the two can be compared directly.

Reads JSON and never has a number typed into it, so a re-measurement cannot leave
a stale annotation behind — same rule as ``make_figures.py``, whose palette and
surface this borrows rather than inventing a second look for one project.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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


def optimal_threshold(values: np.ndarray, other: float, penalty: float, gamma: float) -> np.ndarray:
    """Extra steps worth walking for the richer objective, under discounting.

    Setting the two discounted returns equal gives g^D (c + other) = c + v with
    c = penalty/(1-g). The distance already walked cancels, so the threshold
    depends on the two values alone and not on how far away anything is.
    """
    c = penalty / (1 - gamma)
    return np.log((c + values) / (c + other)) / np.log(gamma)


def band(ax, points, colour, label, marker="o"):
    v = np.array([p["value"] for p in points])
    s = np.array([p["steps"] for p in points])
    lo = np.array([p["lo"] for p in points])
    hi = np.array([p["hi"] for p in points])
    ax.fill_between(v, lo, hi, color=colour, alpha=0.18, lw=0, zorder=2)
    ax.plot(v, s, marker, color=colour, ms=4.2, lw=1.6, ls="-", mec=SURFACE, mew=0.7, zorder=3, label=label)
    return v, s


def main() -> None:
    d = json.loads((DATA / "wide_value_axis.json").read_text())
    entries = [e for e in d["series"] if e.get("trained")]
    if not entries:
        raise SystemExit("no series with trained arms — run figures/extract_wide_axis.py first")

    fig, axes = plt.subplots(1, len(entries), figsize=(5.6 * len(entries), 4.2), squeeze=False, sharey=True)

    for ax, entry in zip(axes[0], entries):
        other = d["other_objective"][entry["sweep"]]
        trained = entry["trained"]
        v = np.array([p["value"] for p in trained])

        grid = np.linspace(v.min() - 0.02, v.max() + 0.02, 200)
        ax.plot(grid, optimal_threshold(grid, other, d["step_penalty"], d["discount"]),
                color=MUTED, lw=1.1, ls="--", zorder=1)
        ax.annotate("optimal", xy=(grid[14], optimal_threshold(grid, other, d["step_penalty"], d["discount"])[14]),
                    xytext=(6, 4), textcoords="offset points", color=MUTED, fontsize=8)

        base = d["base_exchange_rate"].get(entry["agent"])
        if base is not None:
            ax.axhline(base, color=MUTED, lw=0.7, ls=":", zorder=0)

        band(ax, trained, BLUE, "fine-tuned on that value")
        if entry.get("written_heldout"):
            band(ax, entry["written_heldout"], ORANGE, "written, that arm held out", marker="s")

        # The old grid, for scale: seven arms over 0.3 to 0.9. Labelled along the
        # bottom so it cannot collide with the legend or the curves.
        ax.axvspan(0.3, 0.9, color=MUTED, alpha=0.10, lw=0, zorder=0)
        ax.annotate("the old seven-arm grid", xy=(0.6, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", color=MUTED, fontsize=8)

        n = entry["n"]
        ax.set_title(f"{entry['agent']}  ·  sweep {entry['sweep']}  ·  {n.get('trained', 0)} arms", color=INK)
        ax.set_xlabel(f"what the swept objective is worth  (the other pays {other:g})")
        ax.grid(axis="y", color=MUTED, alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)

    axes[0][0].set_ylabel("extra steps walked for the richer objective")
    axes[0][0].legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.0, 0.08))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig13_wide_threshold.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / 'fig13_wide_threshold.png'}")

    for entry in entries:
        t = entry["trained"]
        print(f"  {entry['agent']} {entry['sweep']}: values {t[0]['value']:.2f}-{t[-1]['value']:.2f}, "
              f"steps {t[-1]['steps']:.1f}-{t[0]['steps']:.1f}, "
              f"min reach {min(p['reach'] for p in t):.1f}%")


if __name__ == "__main__":
    main()
