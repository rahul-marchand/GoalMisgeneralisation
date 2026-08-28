"""Flip points for bcnv11: each level's own gap reading, from the decoded grid.

    uv run python scripts/flip_points_bc.py | tee results/flip-points-bcnv11.txt

Orders every decoded model of the value-axis grid by its measured crossing and
fits one step per level (:mod:`goalmisgen.analysis.flips`). The step's location
is the level's effective gap as the model computes it - ``gap * (1 + eps)`` in
UtilityRule.md's part-3 rule - measured behaviourally, with no probe involved.

Writes per-seed targets to ``figures/data/h1/flip/bcnv11.s<seed>.npz`` so the
activation probes can be scored against the model's own estimate rather than
the truth. Levels are the first 20,000 of ``offline/demos/test.rho100``, the
prefix every grid decode shares.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.analysis.flips import step_fit
from goalmisgen.provenance import header

ROOT = Path(__file__).resolve().parent.parent
OFFSETS = (-0.45, -0.40, -0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45)
N = 20_000
BAND = (4, 18)
"""True gaps the grid brackets well; outside it most flips are censored."""


def load(path: Path):
    z = np.load(path)
    return (
        z["reached"][:N],
        z["d_rich"][:N].astype(int) - z["d_poor"][:N].astype(int),
        z["reached_fid"][:N].astype(int) == z["colour_of_rich"][:N].astype(int),
    )


def crossing(reached, gap, took) -> float:
    """Where the binned take-richer rate crosses one half - read, not fitted."""
    gap, took = gap[reached], took[reached].astype(float)
    grid = np.arange(-25, 60)
    rate = np.array([took[gap == g].mean() if (gap == g).sum() >= 25 else np.nan for g in grid])
    good = np.isfinite(rate)
    xs, ys = grid[good], rate[good]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            return float(xs[i] + (ys[i] - 0.5) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i]))
    return float("nan")


def model_files(data: Path, seed: int):
    yield data / f"base.s{seed}.npz"
    for folder in ("arms", "written"):
        for sweep in ("o0", "o1"):
            for offset in OFFSETS:
                tag = f"{sweep}{'-' if offset < 0 else '+'}{abs(round(offset * 100)):03d}"
                path = data / f"{folder}.s{seed}" / f"{tag}.npz"
                if path.exists():
                    yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "figures" / "data" / "h1" / "grid")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "data" / "h1" / "flip")
    args = parser.parse_args()

    print(header())
    print()
    print(f"First {N:,} levels of offline/demos/test.rho100; unreached counts as not-taken.")
    print(f"Models are kept when their crossing lies in (0.5, 25) steps.\n")

    rng = np.random.default_rng(0)
    residuals = {}
    for seed in (1, 2, 3):
        thetas, chosen, rates, gap = [], [], [], None
        for path in model_files(args.data, seed):
            reached, this_gap, took = load(path)
            theta = crossing(reached, this_gap, took)
            if not np.isfinite(theta) or not (0.5 < theta < 25):
                continue
            thetas.append(theta)
            chosen.append(took)
            gap = this_gap if gap is None else gap
            # The model's take rate at each integer gap (nearest bin where thin),
            # for the no-stable-gap null below.
            grid = np.arange(-25, 60)
            rate = np.array([took[this_gap == g].mean() if (this_gap == g).sum() >= 25 else np.nan for g in grid])
            good = np.flatnonzero(np.isfinite(rate))
            nearest = good[np.abs(grid[good][None, :] - grid[:, None]).argmin(axis=1)]
            rates.append(rate[nearest][np.clip(this_gap - grid[0], 0, len(grid) - 1)])
        order = np.argsort(thetas)
        result = step_fit(np.asarray(thetas)[order], np.stack(chosen)[order])

        band = result.bracketed & (gap >= BAND[0]) & (gap <= BAND[1])
        res = result.flip - gap
        q = np.nanpercentile(res[band], [25, 50, 75])
        eps = result.flip[band] / gap[band] - 1
        print(f"s{seed}: {len(result.thetas)} models, theta {result.thetas.min():.1f}..{result.thetas.max():.1f}")
        print(
            f"  bracketed {result.bracketed.mean():.1%}"
            f"  (censored low {result.censored_low.mean():.1%}, high {result.censored_high.mean():.1%})"
        )
        # Step cleanliness is only meaningful where the step is actually tested:
        # censored levels are clean trivially. Judge it in the band, against a
        # null where each model keeps its curve but levels keep nothing.
        in_band = (gap >= BAND[0]) & (gap <= BAND[1])
        M = len(result.thetas)
        real = step_fit(result.thetas, np.stack(chosen)[order][:, in_band])
        R = np.stack(rates)[order][:, in_band]
        null = step_fit(result.thetas, rng.random(R.shape) < R)
        print(
            f"  violations in gap band: per-decision {real.violations.mean() / M:.1%}"
            f"  (zero {np.mean(real.violations == 0):.1%}, <=2 {np.mean(real.violations <= 2):.1%})"
        )
        print(
            f"  no-stable-gap null:     per-decision {null.violations.mean() / M:.1%}"
            f"  (zero {np.mean(null.violations == 0):.1%}, <=2 {np.mean(null.violations <= 2):.1%})"
        )
        print(f"  gap {BAND[0]}..{BAND[1]} ({band.sum():,} levels):")
        print(f"    flip - gap  median {q[1]:+.2f}, IQR {q[2] - q[0]:.2f} steps   [part 1 within-cell width 4.0..5.4]")
        print(f"    eps         median {np.median(eps):+.3f}, IQR {np.percentile(eps, 75) - np.percentile(eps, 25):.3f}")

        args.out.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out / f"bcnv11.s{seed}.npz",
            thetas=result.thetas,
            flip=result.flip,
            violations=result.violations,
            censored_low=result.censored_low,
            censored_high=result.censored_high,
            gap=gap,
        )
        print(f"  saved {args.out / f'bcnv11.s{seed}.npz'}\n")
        residuals[seed] = np.where(band, res, np.nan)

    print("cross-seed correlation of flip - gap (common bracketed levels in the band):")
    for a, b in ((1, 2), (1, 3), (2, 3)):
        both = np.isfinite(residuals[a]) & np.isfinite(residuals[b])
        r = np.corrcoef(residuals[a][both], residuals[b][both])[0, 1]
        print(f"  s{a}-s{b}: {r:+.3f}  (n {both.sum():,})")
    print("The binary within-cell xi gave +0.26..+0.35; a higher value here means the")
    print("continuous flip point recovers more of the same per-level misread.")


if __name__ == "__main__":
    main()
