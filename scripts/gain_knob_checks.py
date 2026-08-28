"""Red-team part 3: is the stretch or the exponential a measurement artifact?

    uv run python scripts/gain_knob_checks.py | tee results/utility-rule-gain-checks-bcnv11.txt

Four ways the result could be fake, each checked against the same grid:

1. Ceiling: the top writes sit against the maze's expressible range, so refit
   the log-linear axis excluding every model with theta > 20.
2. Method: the first crossing of a noisy, non-monotone binned curve is biased
   early, so recompute with isotonic (PAVA) rates, and with bin floors of 25
   and 100 levels.
3. Composition: levels at large gaps are a different population (bounded
   mazes force d_poor down), so recompute the collapse holding d_poor to a
   fixed band.
4. Range: report where measurable data actually ends, to mark which crossings
   are censored.
"""

from pathlib import Path

import numpy as np

from goalmisgen.provenance import header

DATA = Path(__file__).resolve().parent.parent / "figures" / "data" / "h1" / "grid"
OFFSETS = (-0.45, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45)
SEEDS = (1, 2, 3)


def tag(sweep, offset):
    return f"{sweep}{'-' if offset < 0 else '+'}{abs(round(offset * 100)):03d}"


def path_of(folder, seed, sweep, offset):
    return DATA / f"base.s{seed}.npz" if offset == 0.0 else DATA / f"{folder}.s{seed}" / f"{tag(sweep, offset)}.npz"


def binned(path, min_count, d_poor_band=None):
    z = np.load(path)
    m = z["reached"]
    d_rich, d_poor = z["d_rich"].astype(int), z["d_poor"].astype(int)
    if d_poor_band is not None:
        m = m & (d_poor >= d_poor_band[0]) & (d_poor <= d_poor_band[1])
    gap = (d_rich - d_poor)[m]
    took = (z["reached_fid"].astype(int) == z["colour_of_rich"].astype(int))[m].astype(float)
    grid = np.arange(-25, 60)
    counts = np.array([(gap == g).sum() for g in grid])
    rate = np.array([took[gap == g].mean() if c >= min_count else np.nan for g, c in zip(grid, counts)])
    good = np.isfinite(rate)
    return grid[good], rate[good], counts[good]


def pava_decreasing(y, w):
    """Weighted least-squares fit, constrained non-increasing."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); sizes.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            vals[-2:] = [v]; wts[-2:] = [wts[-2] + wts[-1]]; sizes[-2:] = [sizes[-2] + sizes[-1]]
    return np.concatenate([np.full(n, v) for v, n in zip(vals, sizes)])


def crossing(xs, ys, level):
    for i in range(len(xs) - 1):
        if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
            return xs[i] + (ys[i] - level) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i])
    return float("nan")


def quartiles(path, min_count=25, iso=False, d_poor_band=None):
    xs, ys, counts = binned(path, min_count, d_poor_band)
    if iso and len(ys):
        ys = pava_decreasing(ys, counts)
    return tuple(crossing(xs, ys, lv) for lv in (0.5, 0.75, 0.25))


def r2_and_slope(x, y):
    c = np.polyfit(x, y, 1)
    return 1 - (y - np.polyval(c, x)).var() / np.var(y), c[0]


print(header())
print("bcnv11 value-axis grid, robustness checks behind UtilityRule.md part 3.\n")
print("== 1. the exponential, with and without the ceiling (theta > 20 dropped) ==")
print(f"{'series':<14} {'c all':>6} {'R2 all':>7} | {'c <=20':>6} {'R2 <=20':>8} {'n kept':>7}")
for sweep in ("o0", "o1"):
    sign = 1 if sweep == "o0" else -1
    for seed in SEEDS:
        offs, ths = [], []
        for off in OFFSETS:
            th, _, _ = quartiles(path_of("written", seed, sweep, off))
            if np.isfinite(th) and th > 0:
                offs.append(off); ths.append(th)
        offs, ths = np.array(offs), np.array(ths)
        r2a, ca = r2_and_slope(offs, np.log(ths))
        keep = ths <= 20
        r2s, cs = r2_and_slope(offs[keep], np.log(ths[keep]))
        print(f"written {sweep} s{seed} {sign*ca:>6.2f} {r2a:>7.4f} | {sign*cs:>6.2f} {r2s:>8.4f} {int(keep.sum()):>7}")

print("\n== 2. collapse stats under different measurement choices ==")
print("mean +- within-series sd of q25/theta and q75/theta over the 12 series")
for label, kw, cap in [
    ("first-crossing, bins>=25 (doc)", dict(min_count=25), None),
    ("first-crossing, bins>=100", dict(min_count=100), None),
    ("isotonic, bins>=25", dict(min_count=25, iso=True), None),
    ("isotonic, bins>=100", dict(min_count=100, iso=True), None),
    ("first-crossing, theta<=17 only", dict(min_count=25), 17.0),
]:
    lo_stats, hi_stats = [], []
    for folder in ("arms", "written"):
        for sweep in ("o0", "o1"):
            for seed in SEEDS:
                lo, hi = [], []
                for off in OFFSETS:
                    th, q25, q75 = quartiles(path_of(folder, seed, sweep, off), **kw)
                    if not (np.isfinite(th) and th > 1 and np.isfinite(q25) and np.isfinite(q75)):
                        continue
                    if cap is not None and th > cap:
                        continue
                    lo.append(q25 / th); hi.append(q75 / th)
                if len(lo) >= 5:
                    lo_stats.append((np.mean(lo), np.std(lo))); hi_stats.append((np.mean(hi), np.std(hi)))
    lm, ls = np.array(lo_stats).mean(0)
    hm, hs = np.array(hi_stats).mean(0)
    print(f"  {label:<34} q25/t {lm:.2f} ± {ls:.3f}    q75/t {hm:.2f} ± {hs:.3f}   ({len(lo_stats)} series)")

print("\n== 3. composition: hold d_poor in a fixed band (5..12), written o0 s1 ==")
print(f"{'offset':>7} {'theta':>6} {'q25/t':>6} {'q75/t':>6}    (all-level values for comparison)")
for off in (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20):
    th, q25, q75 = quartiles(path_of("written", 1, "o0", off), d_poor_band=(5, 12))
    tha, q25a, q75a = quartiles(path_of("written", 1, "o0", off))
    band = f"{q25/th:>6.2f} {q75/th:>6.2f}" if np.isfinite(th) and np.isfinite(q25) and np.isfinite(q75) else "   nan    nan"
    alla = f"{q25a/tha:.2f} / {q75a/tha:.2f}" if np.isfinite(tha) and np.isfinite(q25a) and np.isfinite(q75a) else "nan"
    print(f"{off:>7.2f} {th:>6.2f} {band}    ({alla})")

print("\n== 4. deep writes: does log-linearity continue below -0.45? ==")
print("written far outside the trained range; exp prediction uses c fitted on |offset| <= 0.45")
DEEP = DATA.parent / "deep"
C = {("o0", 1): 2.31, ("o0", 2): 1.94, ("o0", 3): 2.13, ("o1", 1): -2.26}
BASE_T = {1: 10.33, 2: 10.34, 3: 10.35}
print(f"{'series':<8} {'offset':>7} {'theta':>7} {'exp pred':>9} {'reached':>8}")
for sweep, seed in (("o0", 1), ("o0", 2), ("o0", 3), ("o1", 1)):
    for off in (-0.55, -0.65, -0.80, -1.00, -1.20, -1.50, -2.00):
        signed = off if sweep == "o0" else -off
        path = DEEP / f"s{seed}" / f"{sweep}{'-' if signed < 0 else '+'}{abs(round(signed * 100)):03d}.npz"
        if not path.exists():
            continue
        th, q25, q75 = quartiles(path)
        z = np.load(path)
        pred = BASE_T[seed] * np.exp(C[(sweep, seed)] * signed)
        print(f"{sweep} s{seed}  {signed:>7.2f} {th:>7.2f} {pred:>9.2f} {z['reached'].mean():>8.1%}")
    print()

print("\n== 5. single-block writes: does the move factorise across layers? ==")
print("axis written one parameter block at a time; c_g from log theta vs offset")
MASKED = DATA.parent / "masked"
GROUPS = ("embed", "block_0", "block_1", "block_2", "block_3", "head")
M_OFFSETS = (-0.45, -0.20, 0.20, 0.45)
for seed in SEEDS:
    base_theta, _, _ = quartiles(DATA / f"base.s{seed}.npz")
    rates = {}
    print(f"s{seed}  (base theta {base_theta:.2f})")
    for group in GROUPS:
        thetas = []
        for off in M_OFFSETS:
            path = MASKED / f"s{seed}" / f"{group}.{tag('o0', off)}.npz"
            thetas.append(quartiles(path)[0] if path.exists() else float("nan"))
        xs = np.array([*M_OFFSETS, 0.0])
        ys = np.array([*thetas, base_theta])
        ok = np.isfinite(ys) & (ys > 0)
        rates[group] = np.polyfit(xs[ok], np.log(ys[ok]), 1)[0] if ok.sum() >= 3 else float("nan")
        print(f"  {group:<10} " + " ".join(f"{v:6.2f}" for v in thetas) + f"   c_g {rates[group]:5.2f}")
    print(f"  sum c_g {sum(rates.values()):.2f}")
    for off in M_OFFSETS:
        parts = 0.0
        for group in GROUPS:
            path = MASKED / f"s{seed}" / f"{group}.{tag('o0', off)}.npz"
            th = quartiles(path)[0] if path.exists() else float("nan")
            if np.isfinite(th) and th > 0:
                parts += np.log(th / base_theta)
        full_path = DATA / f"written.s{seed}" / f"{tag('o0', off)}.npz"
        full = np.log(quartiles(full_path)[0] / base_theta) if full_path.exists() else float("nan")
        print(f"  offset {off:>6.2f}: sum of block dlog {parts:>7.3f}   full write {full:>7.3f}")
    cmax = max(rates.values())
    print(f"  c_max {cmax:.2f} -> first factor zeroes near offset {-1 / cmax:.2f}\n")

print("\n== 6. where does measurable data end? ==")
xs, ys, counts = binned(path_of("written", 1, "o0", 0.45), 25)
print(f"top write s1 o0+045: last gap with >=25 levels: {xs.max()}, rate there {ys[-1]:.2f}")
xs, ys, counts = binned(DATA / "base.s1.npz", 25)
print(f"base s1: last gap with >=25 levels: {xs.max()}")
