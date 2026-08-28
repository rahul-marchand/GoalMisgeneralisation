"""What the route model's choice actually depends on, beyond the distance gap.

    uv run python scripts/utility_rule_report.py > results/utility-rule-bcnv11.txt

Reads the per-level decodes and level features in ``figures/data/h1`` and emits
every number behind ``figures/fig_distance_rule.png``. Nothing here touches a
checkpoint; ``scripts/decode_h1.py`` produced the decodes on the data volume.

A *cell* is an exact pair ``(d_rich, d_poor)``. Grouping by the cell rather than
by the gap is what separates the two questions: whether the gap is a sufficient
summary of the two distances, and whether the two distances are sufficient at
all. Within a cell both are fixed by construction, so anything that predicts the
choice there is a dependence on something else entirely.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from goalmisgen.analysis.behaviour import fit_logistic
from goalmisgen.provenance import header

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data" / "h1"
SEEDS = (1, 2, 3)
MIN_CELL = 40
LOGISTIC_SD = np.pi / np.sqrt(3)
"""Standard deviation of a standard logistic: a slope of B per step means the
threshold varies by LOGISTIC_SD / B steps from level to level."""


def load(seed):
    z = np.load(DATA / f"bcnv11.s{seed}.npz")
    took = (z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int)).astype(float)
    return z["d_rich"].astype(int), z["d_poor"].astype(int), took, z["reached"]


def cells(d_rich, d_poor):
    return np.unique(d_rich * 1000 + d_poor, return_inverse=True)[1]


def demean(x, cell):
    counts = np.bincount(cell, minlength=cell.max() + 1)
    means = np.bincount(cell, weights=x, minlength=cell.max() + 1) / np.maximum(counts, 1)
    return x - means[cell]


def within_cell_r2(features, keep, cell, centred):
    """Variance of the within-cell residual explained by a set of level features."""
    x = np.stack([f[keep] for f in features], axis=1).astype(float)
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    x = np.stack([demean(x[:, k], cell) for k in range(x.shape[1])], axis=1)
    beta, *_ = np.linalg.lstsq(x, centred, rcond=None)
    return 1 - ((centred - x @ beta) ** 2).sum() / (centred * centred).sum()


def main() -> None:
    print(header())
    print()
    print("bcnv11, 50,000 held-out levels of offline/demos/test.rho100, greedy decode.")
    print("Expert threshold is exactly 10 steps: (1.0 - 0.5) / 0.05, undiscounted.")

    print("\n=== 1. the threshold, and its drift with absolute distance ===")
    print(f"  {'seed':<6}{'theta at d_poor=0':>19}{'drift per step':>16}{'crossing':>10}")
    for seed in SEEDS:
        d_rich, d_poor, took, reached = load(seed)
        gap = (d_rich - d_poor)[reached]
        xs, thetas = [], []
        for j in range(20):
            m = d_poor[reached] == j
            if m.sum() < 250:
                continue
            w, mu, sd = fit_logistic(gap[m][:, None].astype(float), took[reached][m], steps=6000, lr=0.5, l2=1e-6)
            if abs(w[0]) > 1e-9:
                xs.append(j)
                thetas.append(mu[0] + sd[0] * (-w[1] / w[0]))
        slope, intercept = np.polyfit(xs, thetas, 1)
        w, mu, sd = fit_logistic(gap[:, None].astype(float), took[reached], steps=8000, lr=0.5, l2=1e-6)
        print(f"  s{seed:<5}{intercept:>19.2f}{slope:>16.3f}{mu[0] + sd[0] * (-w[1] / w[0]):>10.2f}")
    print("  A threshold that depends only on the gap would have zero drift.")

    print("\n=== 2. how wide the threshold is, and what the drift accounts for ===")
    print(f"  {'seed':<6}{'pooled':>9}{'within cell':>13}{'drift':>8}{'drift share':>13}")
    for seed in SEEDS:
        d_rich, d_poor, took, reached = load(seed)
        gap, y = (d_rich - d_poor)[reached], took[reached]
        w, _, sd = fit_logistic(gap[:, None].astype(float), y, steps=8000, lr=0.5, l2=1e-6)
        pooled = LOGISTIC_SD / abs(w[0] / sd[0])
        slopes, thetas, sizes = [], [], []
        for j in range(1, 20):
            m = d_poor[reached] == j
            if m.sum() < 400:
                continue
            w2, mu2, sd2 = fit_logistic(gap[m][:, None].astype(float), y[m], steps=8000, lr=0.5, l2=1e-6)
            if abs(w2[0]) > 1e-9:
                slopes.append(abs(w2[0] / sd2[0]))
                thetas.append(mu2[0] + sd2[0] * (-w2[1] / w2[0]))
                sizes.append(float(m.sum()))
        within = LOGISTIC_SD / np.average(slopes, weights=sizes)
        thetas = np.asarray(thetas)
        drift = np.sqrt(np.average((thetas - np.average(thetas, weights=sizes)) ** 2, weights=sizes))
        print(f"  s{seed:<5}{pooled:>9.2f}{within:>13.2f}{drift:>8.2f}{drift**2 / pooled**2:>13.1%}")
    print("  Steps. 'within cell' is the spread left once both distances are held fixed.")

    print("\n=== 3. is the within-cell residual a property of the level or of the network? ===")
    d_rich, d_poor, _, _ = load(1)
    every = np.logical_and.reduce([load(s)[3] for s in SEEDS])
    cell_all = cells(d_rich, d_poor)
    counts = np.bincount(cell_all[every], minlength=cell_all.max() + 1)
    keep = every & (counts[cell_all] >= MIN_CELL)
    cell = np.unique(cell_all[keep], return_inverse=True)[1]
    residual = {s: demean(load(s)[2][keep], cell) for s in SEEDS}
    print(f"  {int(keep.sum()):,} levels decoded by every seed, in {cell.max() + 1} cells (n >= {MIN_CELL})")
    for a, b in itertools.combinations(SEEDS, 2):
        print(f"    corr(residual s{a}, residual s{b}) = {np.corrcoef(residual[a], residual[b])[0, 1]:+.4f}")
    rng = np.random.default_rng(0)
    order = np.argsort(cell)
    bounds = np.searchsorted(cell[order], np.arange(cell.max() + 2))
    null = []
    for _ in range(200):
        shuffled = residual[2][order].copy()
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi - lo > 1:
                rng.shuffle(shuffled[lo:hi])
        null.append(np.corrcoef(residual[1][order], shuffled)[0, 1])
    null = np.asarray(null)
    print(f"    within-cell shuffle null: mean {null.mean():+.4f}, sd {null.std():.4f}")
    print("  Independently trained networks depart from the cell majority on the same levels.")

    print("\n=== 4. what hand-built level features explain of that residual ===")
    geometry = np.load(DATA / "features.npz")
    routes = np.load(DATA / "route_feats.npz")
    geo = [geometry[k] for k in geometry.files] + [
        geometry["l2_rich"] - geometry["l2_poor"],
        geometry["man_rich"] - geometry["man_poor"],
    ]
    rte = [routes[k] for k in routes.files] + [
        routes["turns_rich"] - routes["turns_poor"],
        routes["back_rich"] - routes["back_poor"],
        routes["branch_rich"] - routes["branch_poor"],
    ]
    print(f"  {'seed':<6}{'geometry':>10}{'route shape':>13}{'both':>8}{'random':>9}")
    for seed in SEEDS:
        d_rich, d_poor, took, reached = load(seed)
        c_all = cells(d_rich, d_poor)
        n = np.bincount(c_all, minlength=c_all.max() + 1)
        hits = np.bincount(c_all, weights=took, minlength=c_all.max() + 1)
        # Only cells that saw both outcomes carry any within-cell variance to explain.
        informative = (n >= MIN_CELL) & (hits >= 2) & (n - hits >= 2)
        k = reached & informative[c_all]
        c = np.unique(c_all[k], return_inverse=True)[1]
        centred = demean(took[k], c)
        rng = np.random.default_rng(1)
        noise = [rng.standard_normal(len(took)) for _ in range(len(geo) + len(rte))]
        print(
            f"  s{seed:<5}{within_cell_r2(geo, k, c, centred):>10.4f}"
            f"{within_cell_r2(rte, k, c, centred):>13.4f}"
            f"{within_cell_r2(geo + rte, k, c, centred):>8.4f}"
            f"{within_cell_r2(noise, k, c, centred):>9.4f}"
        )
    print("  21 features against a same-sized random control. The shared component in (3)")
    print("  is an order of magnitude larger than anything these explain.")


if __name__ == "__main__":
    main()
