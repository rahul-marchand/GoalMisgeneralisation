"""Does writing the value axis move the threshold only, or its shape as well?

    uv run python scripts/arm_threshold_shape.py [ARM_DIR]

``027`` measures each arm's exchange rate and its competence -- both means. This
asks the question those cannot: as the axis sweeps the threshold from about 2 to
about 23 steps, does the *spread* of the threshold stay put, and do the same
mazes stay hard?

Two circuits predict identical results in ``027``. In one the model holds a
number already denominated in steps and compares the distance gap against it;
its noise is in steps too, so sigma_theta is invariant however far theta moves.
In the other it compares a worth-like quantity w against a gain-scaled distance
g * dhat, making theta = w/g a ratio that is never an activation; noise on that
comparison reaches steps by dividing by g, so sigma_theta tracks theta.

Fitting sigma = sqrt(c^2 + (k*theta)^2) separates them: the floor c is whatever
noise is already in steps, and a non-zero k is the ratio circuit. It does not
say whether the axis moved w or g -- those differ by a common rescaling of both
sides, which is invisible to behaviour.

Reads the per-arm decodes written by ``scripts/decode_h1.py``; see
``UtilityRule.md`` for the base-model measurement this extends.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from goalmisgen.analysis.behaviour import fit_logistic
from goalmisgen.provenance import header

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "figures" / "data" / "h1" / "bcnv11.s1.npz"
MIN_CELL = 40
LOGISTIC_SD = np.pi / np.sqrt(3)


def load(path):
    z = np.load(path)
    took = (z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int)).astype(float)
    return z["d_rich"].astype(int), z["d_poor"].astype(int), took, z["reached"]


def curve(gap, took):
    """Crossing and spread of one model's psychometric curve, in steps."""
    w, mu, sd = fit_logistic(gap[:, None].astype(float), took, steps=8000, lr=0.5, l2=1e-6)
    slope = w[0] / sd[0]
    if abs(slope) < 1e-9:
        return float("nan"), float("nan")
    return float(mu[0] + sd[0] * (-w[1] / w[0])), float(LOGISTIC_SD / abs(slope))


def residual(took, reached, cell_all, informative):
    keep = reached & informative[cell_all]
    cell = np.unique(cell_all[keep], return_inverse=True)[1]
    counts = np.bincount(cell, minlength=cell.max() + 1)
    means = np.bincount(cell, weights=took[keep], minlength=cell.max() + 1) / np.maximum(counts, 1)
    return keep, took[keep] - means[cell]


def offset_of(path: Path) -> float:
    """``o0-045`` -> -0.45. The sign is the character before the three digits."""
    tag = path.stem.split("o0")[-1]
    # Hundredths, signed: the inverse of goalmisgen.volume.offset_tag.
    return int(tag) / 100 if len(tag) == 4 else float("nan")


def main() -> None:
    arm_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "data" / "h1" / "arms"
    print(header())
    print()
    print("bcnv11.s1, o0 sweep. Every model decoded on the same 50,000 held-out")
    print("levels of test.rho100, at the BASE values, so the exchange rates are comparable.")

    d_rich, d_poor, took_b, reached_b = load(BASE)
    cell_all = np.unique(d_rich * 1000 + d_poor, return_inverse=True)[1]
    gap = d_rich - d_poor

    def informative_for(took, reached):
        n = np.bincount(cell_all[reached], minlength=cell_all.max() + 1)
        hits = np.bincount(cell_all[reached], weights=took[reached], minlength=cell_all.max() + 1)
        return (n >= MIN_CELL) & (hits >= 2) & (n - hits >= 2)

    info_b = informative_for(took_b, reached_b)
    theta_b, sigma_b = curve(gap[reached_b], took_b[reached_b])
    print(f"\nbase: theta {theta_b:.2f}  sigma_theta {sigma_b:.2f}  ratio {sigma_b / theta_b:.3f}")

    arms = sorted(arm_dir.glob("o0*.npz"), key=offset_of)
    if not arms:
        sys.exit(f"no arm decodes under {arm_dir}")

    print(f"\n=== {len(arms)} arms ===")
    print(f"  {'offset':>7}{'expert':>8}{'theta':>8}{'sigma':>8}{'sigma/theta':>13}{'resid corr':>12}{'cells':>7}")
    rows = []
    for path in arms:
        off = offset_of(path)
        _, _, took_a, reached_a = load(path)
        both = reached_a & reached_b
        theta, sigma = curve(gap[reached_a], took_a[reached_a])
        # Signature comparison only where BOTH models are off their rails: a cell
        # saturated for one of them contributes no residual to correlate.
        shared = info_b & informative_for(took_a, reached_a)
        keep = both & shared[cell_all]
        if keep.sum() > 500:
            cell = np.unique(cell_all[keep], return_inverse=True)[1]
            counts = np.bincount(cell, minlength=cell.max() + 1)

            def centred(y):
                m = np.bincount(cell, weights=y, minlength=cell.max() + 1) / np.maximum(counts, 1)
                return y - m[cell]

            corr = float(np.corrcoef(centred(took_b[keep]), centred(took_a[keep]))[0, 1])
            ncell = int(cell.max() + 1)
        else:
            corr, ncell = float("nan"), 0
        expert = (0.5 + off) / 0.05
        print(f"  {off:>+7.2f}{expert:>8.1f}{theta:>8.2f}{sigma:>8.2f}{sigma / theta:>13.3f}{corr:>12.3f}{ncell:>7}")
        rows.append((off, theta, sigma, corr))

    off, theta, sigma, corr = (np.array(c, dtype=float) for c in zip(*rows))
    ok = np.isfinite(theta) & np.isfinite(sigma)
    print("\n=== does the spread follow the threshold? ===")
    slope, intercept = np.polyfit(theta[ok], sigma[ok], 1)
    print(
        f"  sigma = {intercept:+.2f} {slope:+.3f} * theta      corr(theta, sigma) = {np.corrcoef(theta[ok], sigma[ok])[0, 1]:+.3f}"
    )
    print("  A threshold stored in steps predicts slope 0 (sigma flat); a ratio predicts growth.")
    print(
        "  A rescaled distance readout predicts slope ~ sigma_base/theta_base =" f" {sigma_b / theta_b:.3f} and intercept ~ 0."
    )
    good = np.isfinite(corr)
    if good.sum():
        print(f"\n  residual signature vs base: {np.nanmin(corr):+.3f} to {np.nanmax(corr):+.3f} over {int(good.sum())} arms")
        print("  (base-to-base is 1.0 by construction; seed-to-seed was +0.26..+0.35)")


if __name__ == "__main__":
    main()
