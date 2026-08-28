"""The threshold is a gain, not a stored count: stretch, collapse, log-linear dial.

    uv run python figures/make_gain_knob.py

Part 3's figure. A stored step-count edited in the weights would slide the
psychometric curve rigidly and move theta linearly in the offset. A gain
stretches the curve about zero (so dividing the gap by each model's own theta
collapses every curve onto one master curve) and moves ``log theta`` linearly.
Panels A/B show the stretch and the collapse on one write series; panel C shows
the log-linear dial on all six.

Crossings and rates are read off binned data, never fitted -- the method note in
``UtilityRule.md`` says why. Numbers behind the figure:
``scripts/gain_knob_report.py``.
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
RAMP = ("#f0a170", "#eb6834", "#cc4f1e", "#a63c15", "#7a2c10")
"""One hue, light to dark with theta; validated ordinal ramp."""
O0, O1 = "#eb6834", "#8a63d2"
SEEDS = (1, 2, 3)
SHOWN = (-0.40, -0.20, 0.0, 0.20, 0.45)
"""Panel A/B offsets: written o0 s1, spanning theta 4.2 to 29."""
OFFSETS = (-0.45, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45)


def tag(sweep, offset):
    return f"{sweep}{'-' if offset < 0 else '+'}{abs(round(offset * 100)):03d}"


def path_of(folder, seed, sweep, offset):
    return DATA / f"base.s{seed}.npz" if offset == 0.0 else DATA / f"{folder}.s{seed}" / f"{tag(sweep, offset)}.npz"


def curve(path):
    z = np.load(path)
    m = z["reached"]
    gap = (z["d_rich"].astype(int) - z["d_poor"].astype(int))[m]
    took = (z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int))[m].astype(float)
    grid = np.arange(-25, 60)
    rate = np.array([took[gap == g].mean() if (gap == g).sum() >= 25 else np.nan for g in grid])
    good = np.isfinite(rate)
    return grid[good], rate[good]


def cross(xs, ys, level=0.5):
    for i in range(len(xs) - 1):
        if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
            return xs[i] + (ys[i] - level) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i])
    return float("nan")


fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.7), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(color=GRID, lw=0.6, zorder=0)

# A and B: the same five written models, raw and rescaled.
for a in ax[:2]:
    a.axhline(0.5, color=INK2, lw=0.8, ls=":", zorder=1)
    a.set_ylim(-0.03, 1.06)
thetas = []
for colour, offset in zip(RAMP, SHOWN):
    xs, ys = curve(path_of("written", 1, "o0", offset))
    theta = cross(xs, ys)
    thetas.append((theta, colour))
    ax[0].plot(xs, ys, "-", color=colour, lw=2, zorder=4)
    ax[1].plot(xs / theta, ys, "-", color=colour, lw=2, zorder=4)
# A column in the panel's empty corner; on the curves themselves the labels collide.
for row, (theta, colour) in enumerate(sorted(thetas, reverse=True)):
    ax[0].text(46, 0.99 - 0.075 * row, f"$\\bar\\theta$ = {theta:.0f}", color=colour, fontsize=9, ha="right", va="top", zorder=6)
ax[0].text(-14.5, 0.10, "each curve: one model\n(base + offset $\\times$ axis),\nsame 20k mazes", color=INK2, fontsize=8.5, va="bottom")
ax[1].axvline(1.0, color=INK2, lw=0.8, ls=":", zorder=1)
ax[0].set_xlim(-16, 48)
ax[0].set_xlabel("$\\Delta d$, extra steps to the richer objective", color=INK2, fontsize=10)
ax[0].set_ylabel("rate of taking the richer objective", color=INK2, fontsize=10)
ax[0].set_title("A  Writing a new $\\bar\\theta$ stretches the curve", color=INK, fontsize=11.5, loc="left", pad=10)
ax[1].set_xlim(-2.1, 3.1)
ax[1].set_xlabel("$\\Delta d\\, /\\, \\bar\\theta$", color=INK2, fontsize=10)
ax[1].set_title("B  Divided by each model's $\\bar\\theta$, one curve", color=INK, fontsize=11.5, loc="left", pad=10)
ax[1].text(1.06, 0.03, "$\\Delta d = \\bar\\theta$", color=INK2, fontsize=8.5)

# C: the axis on a log scale: the six write series in range, and the deep
# writes beyond it, which fall under the extended fit.
A = ax[2]
A.set_yscale("log")
DEEP = DATA.parent / "deep"
DEEP_OFFSETS = (0.55, 0.65, 0.80, 1.00, 1.20)
for sweep, colour in (("o0", O0), ("o1", O1)):
    stack = []
    for seed in SEEDS:
        thetas = []
        for offset in OFFSETS:
            xs, ys = curve(path_of("written", seed, sweep, offset))
            thetas.append(cross(xs, ys))
        stack.append(thetas)
    stack = np.array(stack)
    A.fill_between(OFFSETS, np.nanmin(stack, 0), np.nanmax(stack, 0), color=colour, alpha=0.22, lw=0, zorder=3)
    A.plot(OFFSETS, np.nanmedian(stack, 0), "-o", color=colour, lw=2, ms=4.2, mec=SURF, mew=0.9, zorder=5)
    # The in-range fit, dotted where it is extrapolation.
    sign = -1 if sweep == "o0" else 1
    xx = np.linspace(0.45, 1.30, 40) * sign
    A.plot(xx, 10.34 * np.exp((2.13 if sweep == "o0" else -2.26) * xx), ls=":", lw=1.3, color=colour, zorder=2)
    for seed in SEEDS:
        for magnitude in DEEP_OFFSETS:
            path = DEEP / f"s{seed}" / f"{tag(sweep, sign * magnitude)}.npz"
            if not path.exists():
                continue
            xs, ys = curve(path)
            theta = cross(xs, ys)
            if np.isfinite(theta) and theta > 0.35:
                A.plot(sign * magnitude, theta, "o", ms=4.5, mfc="none", mec=colour, mew=1.3, zorder=5)
A.axvspan(-0.45, 0.45, color=INK2, alpha=0.05, lw=0, zorder=0)
x = np.linspace(-0.45, 0.45, 200)
for expert in (10 + 20 * x, 10 - 20 * x):
    A.plot(x[expert > 0], expert[expert > 0], ls="--", lw=1.5, color=INK2, zorder=2)
A.set_xlim(-1.32, 1.32)
A.set_ylim(0.35, 36)
A.set_yticks([0.5, 1, 2, 5, 10, 20, 30], labels=["0.5", "1", "2", "5", "10", "20", "30"])
A.set_xlabel("offset along the axis", color=INK2, fontsize=10)
A.set_ylabel("$\\bar\\theta$, log scale", color=INK2, fontsize=10)
A.set_title("C  Multiplies $\\bar\\theta$ equally, inside the trained range", color=INK, fontsize=11.5, loc="left", pad=10)
A.text(0.0, 30.5, "trained range", color=INK2, fontsize=8.5, ha="center")
A.text(0.62, 22, "colour 0 raised", color=O0, fontsize=8.5)
A.text(-1.26, 22, "colour 1 raised", color=O1, fontsize=8.5, ha="left")
A.text(-0.43, 0.55, "expert optimum $\\theta^*$", color=INK2, fontsize=8.5)
A.text(-0.42, 7.5, "$\\bar\\theta \\propto e^{2.1\\,\\cdot\\, \\mathrm{offset}}$", color=INK, fontsize=9.5, ha="center")
A.text(-1.26, 0.72, "beyond the range, writes fall under\nthe extended fit; $\\bar\\theta \\to 0$ by $-1.5$", color=INK2, fontsize=8.5, va="bottom")

fig.suptitle(
    "Figure 4 — The threshold is a gain, not a stored count   ·   written models, A/B one series, C all six plus deep writes",
    color=INK,
    fontsize=12,
    x=0.008,
    ha="left",
    y=0.985,
)
fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.13, wspace=0.21)
fig.savefig(ROOT / "figures" / "fig_gain_knob.png", dpi=200, facecolor=SURF)
fig.savefig(ROOT / "figures" / "fig_gain_knob.pdf", facecolor=SURF)
print("saved")
