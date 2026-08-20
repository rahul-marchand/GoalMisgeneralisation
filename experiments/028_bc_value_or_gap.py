"""One knob or two? cos(axis_0, axis_1) for the route model.

    uv run python experiments/028_bc_value_or_gap.py /workspace/data/offline/runs/bcnv11.s1 --steps 1000

The offline twin of ``015_value_or_gap.py``. ``027`` fits one axis per swept
objective. If the model holds a threshold on the difference in distances (one
knob), raising colour 0's value and raising colour 1's move the same weights in
opposite senses: ``cos(axis_0, axis_1)`` about -1. Two value registers need
not be anti-parallel: about 0.

Both axes are noisy, so the raw cosine is attenuated toward zero; the split-half
reliability of each is printed and used for a secondary disattenuated reading,
and the load-bearing statistic is a permutation null (offsets shuffled among
the diffs of one sweep, which keeps the shared drift and destroys the value
signal), as in ``015`` and ``goalmisgen.analysis.weights``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from goalmisgen.analysis.weights import cosine, fit_axis_and_drift, permutation_cosines, permutation_p_value
from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def fit(diffs):
    offsets = np.array(sorted(o for o in diffs if abs(o) > 1e-9))
    stacked = np.stack([diffs[o] for o in offsets])
    axis, drift = fit_axis_and_drift(offsets, stacked)
    halves = (offsets[0::2], offsets[1::2])
    half_axes = [fit_axis_and_drift(np.array(h), np.stack([diffs[o] for o in h]))[0] for h in halves]
    return offsets, stacked, axis, drift, cosine(*half_axes)


def main() -> None:
    args = parse_args()
    print(header())
    print()
    base = load_base(args.run)
    sweeps = {}
    for sweep in ("o0", "o1"):
        arms = arm_dirs(args.run, sweep, args.steps)
        if len(arms) < 4:
            sys.exit(f"sweep {sweep}: only {len(arms)} finished arms under {args.run / 'arms'}")
        sweeps[sweep] = load_diffs(base, arms)
        print(f"{sweep}: {len(arms)} arms")
    o0, s0, axis0, drift0, rel0 = fit(sweeps["o0"])
    o1, s1, axis1, drift1, rel1 = fit(sweeps["o1"])
    observed = cosine(axis0, axis1)

    print("\n=== the two axes ===")
    print(f"  |axis_0| {np.linalg.norm(axis0):.4g}  |axis_1| {np.linalg.norm(axis1):.4g}  |drift_0| {np.linalg.norm(drift0):.4g}  |drift_1| {np.linalg.norm(drift1):.4g}")
    print(f"  cos(drift_0, drift_1) {cosine(drift0, drift1):+.3f}   (the two sweeps' common components; should be alike)")
    print(f"\n  cos(axis_0, axis_1) = {observed:+.3f}")
    print("  about -1: one knob (a threshold on the gap); about 0: two value registers")

    print("\n=== the same gap reached by moving either colour (drift removed) ===")
    print(f"  {'gap change':>11}{'o0 offset':>11}{'o1 offset':>11}{'cos':>8}")
    same_gap = {}
    for o in (0.05, 0.1, 0.2, 0.3, 0.38, 0.4, 0.45):
        for sign in (1, -1):
            a, b = round(sign * o, 10), round(-sign * o, 10)
            if a in sweeps["o0"] and b in sweeps["o1"]:
                c = cosine(sweeps["o0"][a] - drift0, sweeps["o1"][b] - drift1)
                same_gap[f"{a:+.2f}"] = c
                print(f"  {sign * o:>+11.2f}{a:>+11.2f}{b:>+11.2f}{c:>+8.3f}")
    print("  Under one knob these are the same model reached two ways; their diffs should agree.")

    print("\n=== how much of each axis is signal? ===")
    print(f"  split-half reliability  axis_0 {rel0:+.3f}   axis_1 {rel1:+.3f}")
    corrected = observed / np.sqrt(rel0 * rel1) if rel0 > 0 and rel1 > 0 else float("nan")
    print(f"  cos corrected for attenuation {corrected:+.3f}" + ("  (outside [-1, 1]: the correction has broken down)" if abs(corrected) > 1 else ""))

    print("\n=== permutation null ===")
    null = permutation_cosines(o1, s1, axis0, resamples=args.resamples, seed=args.seed)
    p = permutation_p_value(observed, null, alternative="less")
    print(f"  observed {observed:+.3f}; null over {args.resamples} shuffles mean {null.mean():+.3f} sd {null.std():.3f}, 5th pct {np.percentile(null, 5):+.3f}; p(null <= observed) {p:.4f}")
    null_rev = permutation_cosines(o0, s0, axis1, resamples=args.resamples, seed=args.seed + 1)
    p_rev = permutation_p_value(observed, null_rev, alternative="less")
    print(f"  shuffling the other sweep: null mean {null_rev.mean():+.3f} sd {null_rev.std():.3f}; p {p_rev:.4f}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "run": str(args.run), "steps": args.steps, "cos": observed, "reliability_0": rel0, "reliability_1": rel1,
            "cos_disattenuated": corrected, "p_permutation": p, "p_permutation_reverse": p_rev,
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "same_gap": same_gap,
            "axis_norm_0": float(np.linalg.norm(axis0)), "axis_norm_1": float(np.linalg.norm(axis1)),
            "drift_norm_0": float(np.linalg.norm(drift0)), "drift_norm_1": float(np.linalg.norm(drift1)),
            "cos_drifts": cosine(drift0, drift1),
        }, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
