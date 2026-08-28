"""Is the route model's choice a function of the two distances alone?

    uv run python figures/make_distance_rule.py

Reads the per-level decodes in ``figures/data/h1`` (written by
``scripts/decode_h1.py`` against the checkpoints on the data volume, one row
per held-out level) and draws the four panels of ``fig_distance_rule``. No
number is typed into the plot: every annotation is computed from those arrays,
as in ``make_figures.py``.

Panel C is the load-bearing one and is deliberately non-parametric. Fitting one
logistic over the whole range of gaps understates the dependence on absolute
distance, because most levels sit where the choice is saturated; fitting the
50% crossing separately at each ``d_poor`` does not.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from goalmisgen.analysis.behaviour import fit_logistic

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data" / "h1"
FIGS = ROOT / "figures"

SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
S = ["#2a78d6", "#eb6834", "#1baf7a"]
RED = "#e34948"
MID = "#f0efec"
DIV = LinearSegmentedColormap.from_list("d", [RED, MID, S[0]])


def load(p):
    z = np.load(p)
    m = z["reached"]
    return (
        z["d_rich"].astype(int)[m],
        z["d_poor"].astype(int)[m],
        (z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int))[m].astype(float),
    )


dr, dp, y = load(str(DATA / "bcnv11.s1.npz"))
gap, dsum = dr - dp, dr + dp
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.6), facecolor=SURF)
for a in ax.ravel():
    a.set_facecolor(SURF)
    for s in a.spines.values():
        s.set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(color=GRID, lw=0.6, zorder=0)

A = ax[0, 0]
N = 26
P = np.full((N, N), np.nan)
for i in range(1, N):
    for j in range(1, N):
        m = (dr == i) & (dp == j)
        if m.sum() >= 10:
            P[i, j] = y[m].mean()
im = A.imshow(P, origin="lower", cmap=DIV, vmin=0, vmax=1, interpolation="nearest")
A.plot([0, 15], [10, 25], ls="--", lw=1.6, color=INK, zorder=3)
A.text(
    0.97,
    0.05,
    "dashed: expert boundary",
    transform=A.transAxes,
    ha="right",
    fontsize=9,
    color=INK,
    bbox=dict(fc=SURF, ec=GRID, boxstyle="round,pad=0.3"),
)
A.set_xlim(0, N - 1)
A.set_ylim(0, N - 1)
A.set_xlabel("distance to poorer objective", color=INK2, fontsize=10)
A.set_ylabel("distance to richer objective", color=INK2, fontsize=10)
A.set_title("A  Choice over both distances (cells n$\\geq$10)", color=INK, fontsize=11.5, loc="left", pad=10)
cb = fig.colorbar(im, ax=A, fraction=0.046, pad=0.03)
cb.set_label("P(takes richer)", color=INK2, fontsize=9)
cb.ax.tick_params(colors=INK2, labelsize=8)
plt.setp(cb.outline, color=GRID)

B = ax[0, 1]
gs = np.arange(-14, 25)
# The expert flips *at* 10, not between 9 and 10: at a gap of exactly 10 the
# utilities are equal (0.5 - 0.05 x 10 = 0), so solve() calls the level a tie
# and flags it ambiguous. Drawn as an explicit vertical drop at 10 with the tie
# marked, because step(where='mid') would place the drop at 9.5 and read as a
# threshold of 9.5.
B.plot([gs[0], 10, 10, gs[-1]], [1, 1, 0, 0], color=INK2, ls="--", lw=1.6, zorder=2, label="expert (flips at 10)")
B.plot([10], [0.5], "o", ms=5, mfc=SURF, mec=INK2, mew=1.6, zorder=3)
B.annotate(
    "tie at exactly 10",
    xy=(10, 0.5),
    xytext=(14.5, 0.72),
    fontsize=8.5,
    color=INK2,
    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9),
)
for k, seed in enumerate((1, 2, 3)):
    d2, p2, y2 = load(str(DATA / f"bcnv11.s{seed}.npz"))
    g2 = d2 - p2
    curve = [y2[g2 == g].mean() if (g2 == g).sum() >= 25 else np.nan for g in gs]
    B.plot(gs, curve, color=S[k], lw=2, zorder=4, label=f"s{seed}")
    B.plot(gs, curve, "o", ms=3.6, color=S[k], mec=SURF, mew=0.9, zorder=5)
B.axhline(0.5, color=GRID, lw=1, zorder=1)
B.set_xlabel("$\\Delta d$  = (steps to richer) $-$ (steps to poorer)", color=INK2, fontsize=10)
B.set_ylabel("P(takes richer)", color=INK2, fontsize=10)
B.set_title("B  It is a curve, not a step", color=INK, fontsize=11.5, loc="left", pad=10)
lg = B.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
[t.set_color(INK) for t in lg.get_texts()]

C = ax[1, 0]
for k, s in enumerate((1, 2, 3)):
    d2, p2, y2 = load(str(DATA / f"bcnv11.s{s}.npz"))
    g2 = d2 - p2
    xs, cr = [], []
    for j in range(0, 20):
        m = p2 == j
        if m.sum() < 250:
            continue
        w, mu, sd = fit_logistic(g2[m][:, None].astype(float), y2[m], steps=6000, lr=0.5, l2=1e-6)
        if abs(w[0]) > 1e-9:
            xs.append(j)
            cr.append(mu[0] + sd[0] * (-w[1] / w[0]))
    xs, cr = np.array(xs), np.array(cr)
    co = np.polyfit(xs, cr, 1)
    C.plot(xs, cr, "o", ms=5.5, color=S[k], mec=SURF, mew=1.2, zorder=4)
    C.plot(xs, np.polyval(co, xs), color=S[k], lw=2, zorder=3)
    C.text(
        xs[-1] + 0.4, np.polyval(co, xs[-1]), f"s{s}  $+{co[0]:.3f}$", color=S[k], fontsize=9.5, va="center", fontweight="bold"
    )
C.axhline(10, color=INK2, ls="--", lw=1.6, zorder=2)
C.text(7.0, 10.06, "expert: a constant 10 steps", color=INK2, fontsize=9, va="bottom")
C.set_xlim(-0.5, 23.5)
C.set_xlabel("distance to the poorer objective", color=INK2, fontsize=10)
C.set_ylabel("model's threshold $\\theta$ (steps)", color=INK2, fontsize=10)
C.set_title("C  The threshold drifts with absolute distance", color=INK, fontsize=11.5, loc="left", pad=10)

D = ax[1, 1]
for k, s in enumerate((1, 2, 3)):
    d2, p2, y2 = load(str(DATA / f"bcnv11.s{s}.npz"))
    g2 = d2 - p2
    xs, ims = [], []
    for g in range(-12, 25):
        sel = g2 == g
        if sel.sum() < 100:
            continue
        key = d2[sel] * 1000 + p2[sel]
        u, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv)
        pc = np.bincount(inv, weights=y2[sel]) / np.maximum(cnt, 1)
        ok = cnt >= 15
        if ok.sum() < 3:
            continue
        xs.append(g)
        ims.append(np.average(np.minimum(pc, 1 - pc)[ok], weights=cnt[ok]))
    D.plot(xs, ims, "o-", color=S[k], lw=2, ms=5, mec=SURF, mew=1.2)
    D.text(xs[-1] + 0.5, ims[-1], f"s{s}", color=S[k], fontsize=9.5, va="center", fontweight="bold")
D.axhline(0, color=INK2, lw=1.4, ls="--")
D.annotate(
    "a distance-only rule\npredicts zero",
    xy=(-6, 0.0),
    xytext=(-11.5, 0.16),
    fontsize=9,
    color=INK2,
    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9),
)
D.set_xlabel("$\\Delta d$", color=INK2, fontsize=10)
D.set_ylabel("within-cell disagreement", color=INK2, fontsize=10)
D.set_title("D  Identical distances, different choices", color=INK, fontsize=11.5, loc="left", pad=10)

fig.suptitle(
    "Figure 1 — Is the choice a function of the two distances alone?   bcnv11 (0.8M prefix-LM), 50k held-out levels",
    color=INK,
    fontsize=12.5,
    x=0.008,
    ha="left",
    y=0.985,
)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(str(FIGS / "fig_distance_rule.png"), dpi=200, facecolor=SURF)
fig.savefig(str(FIGS / "fig_distance_rule.pdf"), facecolor=SURF)
print("saved")
