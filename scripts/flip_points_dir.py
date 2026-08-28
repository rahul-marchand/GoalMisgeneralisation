"""Flip points from any directory of decoded models, base included.

    uv run python scripts/flip_points_dir.py --base figures/data/scaling/d512l4.npz \
        --arms figures/data/scaling/arms.d512l4 --out figures/data/scaling/flip.d512l4.npz

The generic twin of ``scripts/flip_points_bc.py``: that script walks the
bcnv11 value-axis grid's layout; this one takes a base decode and a directory
of arm decodes in the same five-array schema, wherever they came from. Same
method: order models by their measured crossing, fit one step per level
(:mod:`goalmisgen.analysis.flips`), read each level's effective gap off the
step's location.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.analysis.flips import step_fit
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--theta-range", type=float, nargs=2, default=(0.5, 25.0))
    parser.add_argument("--band", type=int, nargs=2, default=(4, 18))
    return parser.parse_args()


def load(path: Path, n: int):
    z = np.load(path)
    return (
        z["reached"][:n],
        z["d_rich"][:n].astype(int) - z["d_poor"][:n].astype(int),
        z["reached_fid"][:n].astype(int) == z["colour_of_rich"][:n].astype(int),
    )


def crossing(reached, gap, took) -> float:
    gap, took = gap[reached], took[reached].astype(float)
    grid = np.arange(-25, 60)
    rate = np.array([took[gap == g].mean() if (gap == g).sum() >= 25 else np.nan for g in grid])
    good = np.isfinite(rate)
    xs, ys = grid[good], rate[good]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            return float(xs[i] + (ys[i] - 0.5) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i]))
    return float("nan")


def main() -> None:
    args = parse_args()
    print(header())
    print()

    thetas, chosen, gap, names = [], [], None, []
    for path in [args.base, *sorted(args.arms.glob("*.npz"))]:
        reached, this_gap, took = load(path, args.n)
        theta = crossing(reached, this_gap, took)
        if not np.isfinite(theta) or not (args.theta_range[0] < theta < args.theta_range[1]):
            print(f"  dropped {path.name}: crossing {theta}")
            continue
        thetas.append(theta)
        chosen.append(took)
        names.append(path.name)
        gap = this_gap if gap is None else gap
    order = np.argsort(thetas)
    result = step_fit(np.asarray(thetas)[order], np.stack(chosen)[order])

    lo, hi = args.band
    band = result.bracketed & (gap >= lo) & (gap <= hi)
    in_band = (gap >= lo) & (gap <= hi)
    M = len(result.thetas)
    banded = step_fit(result.thetas, np.stack(chosen)[order][:, in_band])
    res = result.flip - gap
    q = np.nanpercentile(res[band], [25, 50, 75])
    print(f"\n{M} models, theta {result.thetas.min():.1f}..{result.thetas.max():.1f}")
    print(f"bracketed {result.bracketed.mean():.1%} (low {result.censored_low.mean():.1%}, high {result.censored_high.mean():.1%})")
    print(
        f"violations in gap band {lo}..{hi}: per-decision {banded.violations.mean() / M:.1%}"
        f"  (zero {np.mean(banded.violations == 0):.1%}, <=2 {np.mean(banded.violations <= 2):.1%})"
    )
    print(f"band flip - gap: median {q[1]:+.2f}, IQR {q[2] - q[0]:.2f} steps")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        thetas=result.thetas,
        flip=result.flip,
        violations=result.violations,
        censored_low=result.censored_low,
        censored_high=result.censored_high,
        gap=gap,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
