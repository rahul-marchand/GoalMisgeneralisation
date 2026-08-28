"""Does the route model's exchange rate depend on absolute distance?

    uv run python scripts/h2_recheck.py [DIR]

Reads the per-level decodes written by ``scripts/decode_h1.py`` (default
``figures/data/h1``) and asks whether the distance *gap* is a sufficient
summary of the choice, or whether the two distances matter beyond their
difference.

Three tests, because the obvious one is misleading. A logistic fitted over the
whole range of gaps reports almost no dependence on ``d_rich + d_poor`` -- but
about four levels in five sit where the choice is saturated, and a single
coefficient estimated mostly from levels that could not have gone either way is
diluted toward zero. The band fit restricts to gaps where the decision is
actually live. The crossing fit is the non-parametric version: fit the 50%
point separately at each ``d_poor`` and read the slope, which is zero if and
only if the gap is sufficient.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

from goalmisgen.analysis.behaviour import fit_logistic


def load(p):
    z = np.load(p)
    m = z["reached"]
    return (
        z["d_rich"].astype(int)[m],
        z["d_poor"].astype(int)[m],
        (z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int))[m].astype(float),
    )


DATA = (
    pathlib.Path(sys.argv[1])
    if len(sys.argv) > 1
    else pathlib.Path(__file__).resolve().parent.parent / "figures" / "data" / "h1"
)

for s in (1, 2, 3):
    dr, dp, y = load(DATA / f"bcnv11.s{s}.npz")
    gap, dsum = dr - dp, dr + dp
    print(f"=== bcnv11.s{s} ===")

    # 1. Where does the 0.5 crossing sit, as a function of d_poor?
    #    H2 (only gap matters) => crossing d_rich = d_poor + theta, slope exactly 1.
    xs, cr = [], []
    for j in range(0, 20):
        m = dp == j
        if m.sum() < 250:
            continue
        g, yy = gap[m], y[m]
        w, mu, sd = fit_logistic(g[:, None].astype(float), yy, steps=6000, lr=0.5, l2=1e-6)
        if abs(w[0]) < 1e-9:
            continue
        xs.append(j)
        cr.append(mu[0] + sd[0] * (-w[1] / w[0]))
    xs, cr = np.array(xs), np.array(cr)
    A = np.polyfit(xs, cr, 1)
    print(f"  crossing gap vs d_poor: slope {A[0]:+.4f}  intercept {A[1]:.2f}   (H2 => slope 0)")
    print("   d_poor:", " ".join(f"{v:>5d}" for v in xs))
    print("   theta :", " ".join(f"{v:>5.1f}" for v in cr))

    # 2. Targeted, well-powered: near-boundary band only
    band = (gap >= 6) & (gap <= 16)
    X = np.stack([gap[band], dsum[band]], 1).astype(float)
    w, mu, sd = fit_logistic(X, y[band], steps=8000, lr=0.5, l2=1e-6)
    print(f"  band gap 6..16 (n={band.sum():,}): coef_gap {w[0]/sd[0]:+.4f}/step   coef_dsum {w[1]/sd[1]:+.4f}/step")
    print(f"    -> threshold shift per +10 steps of d_sum: {-(w[1]/sd[1])/(w[0]/sd[0])*10:+.2f} steps")

    # 3. Same on the FULL range, for contrast with what I reported
    Xf = np.stack([gap, dsum], 1).astype(float)
    wf, muf, sdf = fit_logistic(Xf, y, steps=8000, lr=0.5, l2=1e-6)
    print(f"  full range:  coef_gap {wf[0]/sdf[0]:+.4f}/step   coef_dsum {wf[1]/sdf[1]:+.4f}/step")
    print()
