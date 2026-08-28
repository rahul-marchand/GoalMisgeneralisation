"""The value axis as a dial: one weight direction, and where the threshold lands.

    uv run python figures/make_axis_dial.py

Part 1 established that the model has a threshold in steps. This asks whether
that threshold is *adjustable*: fine-tune at a shifted value (trained), or write
``base + offset * axis`` into the weights (written), and see where it goes.

Crossings are read off binned rates, not fitted. ``indifference_point`` fits a
logistic over the whole range of gaps, most of which is saturated, and that
biases the crossing -- by +0.78 steps on the base model alone. It also returns a
*magnitude*, which is why earlier versions of this plot had to impose the sign
past the preference flip; an empirical crossing keeps its direction.

Colour is method; seeds are a band rather than three more hues. Which seed is
which carries nothing here, only whether they agree, so the band's width is the
replication and the reader is not asked to track six lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data" / "h1" / "grid"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
TRAINED, WRITTEN = "#2a78d6", "#eb6834"
SEEDS = (1, 2, 3)
OFFSETS = (-0.45, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45)
TRUSTED = 0.20
"""Beyond this the written model stops tracking the arm; measured, not assumed."""


def theta(path: Path) -> float:
    """Where the model takes the richer objective half the time, in steps.

    Positive means it walks that many extra steps for colour 0, which is the
    richer objective at the base values every model is scored on.
    """
    if not path.exists():
        return float("nan")
    z = np.load(path)
    m = z["reached"]
    gap = (z["d_rich"].astype(int) - z["d_poor"].astype(int))[m]
    took = (z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int))[m].astype(float)
    grid = np.arange(-20, 50)
    rate = np.array([took[gap == g].mean() if (gap == g).sum() >= 25 else np.nan for g in grid])
    good = np.isfinite(rate)
    xs, ys = grid[good], rate[good]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            return xs[i] + (ys[i] - 0.5) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i])
    return float("nan")


def tag(sweep, offset):
    return f"{sweep}{'-' if offset < 0 else '+'}{abs(round(offset * 100)):03d}"


fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4), facecolor=SURF, sharey=True)
for a in ax:
    a.set_facecolor(SURF)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(color=GRID, lw=0.6, zorder=0)


def band(A, x, series, colour):
    """Median across seeds, with the min-max range behind it."""
    stack = np.vstack(series)
    A.fill_between(x, np.nanmin(stack, 0), np.nanmax(stack, 0), color=colour, alpha=0.22, lw=0, zorder=3)
    A.plot(x, np.nanmedian(stack, 0), "-o", color=colour, lw=2, ms=4.2, mec=SURF, mew=0.9, zorder=5)


base = np.nanmean([theta(DATA / f"base.s{s}.npz") for s in SEEDS])
panels = [
    ("o0", 1.0, "what colour 0 is worth", "colour 1 held at 0.5", 0),
    ("o1", 0.5, "what colour 1 is worth", "colour 0 held at 1.0", 1),
]

for sweep, centre, xlabel, held, col in panels:
    A = ax[col]
    value = np.array([centre + o for o in OFFSETS])
    expert = np.array([((1.0 + o) - 0.5) / 0.05 if sweep == "o0" else (1.0 - (0.5 + o)) / 0.05 for o in OFFSETS])

    A.axvspan(centre - TRUSTED, centre + TRUSTED, color=WRITTEN, alpha=0.055, zorder=0, lw=0)
    A.plot(value, expert, ls="--", lw=1.7, color=INK2, zorder=2)
    A.axhline(base, ls=":", lw=1.2, color=INK2, zorder=1)

    for colour, folder in ((TRAINED, "arms"), (WRITTEN, "written")):
        band(
            A,
            value,
            [np.array([theta(DATA / f"{folder}.s{s}" / f"{tag(sweep, o)}.npz") for o in OFFSETS]) for s in SEEDS],
            colour,
        )

    A.set_xlabel(f"{xlabel}    ({held})", color=INK2, fontsize=10)
    A.set_title(f"{'A' if sweep == 'o0' else 'B'}  Colour {sweep[1]} swept", color=INK, fontsize=11.5, loc="left", pad=10)

ax[0].set_ylabel("$\\bar\\theta$, extra steps walked for colour 0", color=INK2, fontsize=10)
ax[0].set_ylim(-1, 32)
ax[0].text(0.555, base + 0.9, f"base, {base:.1f} steps", color=INK2, fontsize=8.5)
ax[0].text(1.29, 14.2, "optimal", color=INK2, fontsize=9, rotation=21)
ax[1].text(0.20, 11.2, "optimal", color=INK2, fontsize=9, rotation=-21)
handles = [
    plt.Line2D([], [], color=TRAINED, lw=2, marker="o", ms=4, label="trained: one fine-tune per value"),
    plt.Line2D([], [], color=WRITTEN, lw=2, marker="o", ms=4, label="written: base + offset $\\times$ axis"),
    plt.Line2D([], [], color=INK2, lw=1.7, ls="--", label="optimal for those values"),
    mpl.patches.Patch(
        facecolor=WRITTEN,
        alpha=0.18,
        edgecolor="none",
        label=f"shaded: writes match the arms to 0.3 steps (|offset| $\\leq$ {TRUSTED:.1f})",
    ),
]
lg = ax[0].legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
[t.set_color(INK) for t in lg.get_texts()]
# Bands are wider than lines, so the legend needs the room the shading takes.
fig.suptitle(
    "Figure 3 \u2014 One direction in weight space moves the threshold   \u00b7   bcnv11, 3 seeds, 20k held-out levels",
    color=INK,
    fontsize=12,
    x=0.008,
    ha="left",
    y=0.985,
)
fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.125, wspace=0.06)
fig.savefig(ROOT / "figures" / "fig_axis_dial.png", dpi=200, facecolor=SURF)
fig.savefig(ROOT / "figures" / "fig_axis_dial.pdf", facecolor=SURF)
print("saved")
