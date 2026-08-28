"""Part 3 of UtilityRule.md: is the threshold a stored count or a gain?

    uv run python scripts/gain_knob_report.py | tee results/utility-rule-gain-bcnv11.txt

Three witnesses, all read from the decoded grid (``scripts/decode_grid.py``):

1. **Dilation.** Editing a stored step-count slides the psychometric curve
   rigidly (quartile offsets ``q - theta`` invariant); turning a gain stretches
   it about zero (quartile ratios ``q / theta`` invariant).
2. **Dial shape.** A stored count edited linearly in the weights moves theta
   linearly in the offset; a gain moves ``log theta`` linearly.
3. **Nesting.** If each level carries a fixed fractional misread, raising the
   threshold only ever adds levels to the taken set. Backward flips measure the
   fresh idiosyncrasy an edit injects.

Everything is read off binned rates, never fitted -- see the method note in
``UtilityRule.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.provenance import header

ROOT = Path(__file__).resolve().parent.parent
OFFSETS = (-0.45, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45)
SEEDS = (1, 2, 3)


def tag(sweep: str, offset: float) -> str:
    return f"{sweep}{'-' if offset < 0 else '+'}{abs(round(offset * 100)):03d}"


def load(path: Path):
    z = np.load(path)
    reached = z["reached"]
    gap = z["d_rich"].astype(int) - z["d_poor"].astype(int)
    took = z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int)
    return reached, gap, took


def crossings(path: Path, levels=(0.5,)) -> tuple[float, ...]:
    """Where the binned take-richer rate first crosses each level, in steps."""
    reached, gap, took = load(path)
    gap, took = gap[reached], took[reached].astype(float)
    grid = np.arange(-25, 60)
    rate = np.array([took[gap == g].mean() if (gap == g).sum() >= 25 else np.nan for g in grid])
    good = np.isfinite(rate)
    xs, ys = grid[good], rate[good]

    def cross(level: float) -> float:
        for i in range(len(xs) - 1):
            if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
                return xs[i] + (ys[i] - level) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i])
        return float("nan")

    return tuple(cross(level) for level in levels)


def model_path(data: Path, folder: str, seed: int, sweep: str, offset: float) -> Path:
    if offset == 0.0:
        return data / f"base.s{seed}.npz"
    return data / f"{folder}.s{seed}" / f"{tag(sweep, offset)}.npz"


def series(data: Path, folder: str, seed: int, sweep: str):
    """(offset, theta, q25, q75) for every decoded model of one series, base included."""
    rows = []
    for offset in OFFSETS:
        path = model_path(data, folder, seed, sweep, offset)
        if not path.exists():
            continue
        # q25 of the theta distribution is where the rate is still 0.75, and so on.
        theta, q25, q75 = crossings(path, levels=(0.5, 0.75, 0.25))
        if np.isfinite(theta) and theta > 0:
            rows.append((offset, theta, q25, q75))
    return rows


def r_squared(x, y) -> tuple[float, float]:
    coefficients = np.polyfit(x, y, 1)
    residual = y - np.polyval(coefficients, x)
    return 1 - residual.var() / np.var(y), coefficients[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "figures" / "data" / "h1" / "grid")
    args = parser.parse_args()

    print(header())
    print("bcnv11 value-axis grid, 20,000 held-out levels of offline/demos/test.rho100, greedy decode.")
    print("Is the threshold a stored count (curve slides) or a gain (curve stretches)?\n")
    print("== 1. the curve dilates rather than slides ==")
    print("slide predicts q25-t, q75-t invariant; stretch predicts q25/t, q75/t invariant")
    print(f"\nwritten o0 s1:  {'offset':>7} {'theta':>6} | {'q25-t':>6} {'q75-t':>6} | {'q25/t':>6} {'q75/t':>6}")
    for offset, theta, q25, q75 in series(args.data, "written", 1, "o0"):
        print(f"{'':16}{offset:>7.2f} {theta:>6.2f} | {q25 - theta:>6.2f} {q75 - theta:>6.2f} | {q25 / theta:>6.3f} {q75 / theta:>6.3f}")

    stats: dict[str, list] = {"q25-t": [], "q75-t": [], "q25/t": [], "q75/t": []}
    for folder in ("arms", "written"):
        for sweep in ("o0", "o1"):
            for seed in SEEDS:
                rows = [r for r in series(args.data, folder, seed, sweep) if np.isfinite(r[2]) and np.isfinite(r[3])]
                if len(rows) < 5:
                    continue
                _, theta, q25, q75 = map(np.array, zip(*rows))
                for key, values in (("q25-t", q25 - theta), ("q75-t", q75 - theta), ("q25/t", q25 / theta), ("q75/t", q75 / theta)):
                    stats[key].append((values.mean(), values.std()))
    print("\nacross the 12 series (mean of the statistic, then its sd across models):")
    for key, pairs in stats.items():
        means, sds = map(np.array, zip(*pairs))
        print(f"  {key:>6}: mean {means.mean():>7.2f}   within-series sd {sds.mean():.3f}")

    print("\n== 2. the written dial is log-linear:  theta = theta_0 * exp(c * offset) ==")
    print(f"{'series':<16} {'n':>2} {'R2 linear':>10} {'R2 log':>8} {'c':>6}")
    for folder in ("written", "arms"):
        for sweep in ("o0", "o1"):
            sign = 1 if sweep == "o0" else -1  # o1 raises the poorer value, lowering theta
            for seed in SEEDS:
                rows = series(args.data, folder, seed, sweep)
                offsets, thetas = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
                linear, _ = r_squared(offsets, thetas)
                log, slope = r_squared(offsets, np.log(thetas))
                print(f"{folder} {sweep} s{seed:<4} {len(rows):>2} {linear:>10.4f} {log:>8.4f} {sign * slope:>6.2f}")
    print("(arms follow the expert's linear theta* = 10 + 20*offset instead: trained, not written)")

    print("\n== 3. flips are nested: raising theta only adds levels ==")
    print("backward flips (took richer at base, poorer at the write) are forbidden under a fixed per-level misread")
    print(f"{'write':<12} {'theta':>6} {'both reached':>12} {'forward':>8} {'backward':>9} {'back/fwd':>9}")
    for seed in SEEDS:
        reached_b, _, took_b = load(args.data / f"base.s{seed}.npz")
        for offset in (0.05, 0.10, 0.20, 0.30, 0.45):
            path = args.data / f"written.s{seed}" / f"{tag('o0', offset)}.npz"
            if not path.exists():
                continue
            reached_w, _, took_w = load(path)
            n = min(len(reached_b), len(reached_w))  # part 1's base decode is 50k, the grid 20k; same levels, same order
            both = reached_b[:n] & reached_w[:n]
            forward = int((~took_b[:n][both] & took_w[:n][both]).sum())
            backward = int((took_b[:n][both] & ~took_w[:n][both]).sum())
            (theta,) = crossings(path)
            print(f"s{seed} {tag('o0', offset):<9} {theta:>6.2f} {int(both.sum()):>12,} {forward:>8,} {backward:>9,} {backward / max(forward, 1):>9.3f}")


if __name__ == "__main__":
    main()
