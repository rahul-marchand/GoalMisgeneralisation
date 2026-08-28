"""Decode a whole value-axis grid in one process.

    uv run python scripts/decode_grid.py --out /workspace/h1/grid --n 20000

Every model in the grid -- each base, each fine-tuned arm, and each model
written as ``base + offset * axis`` -- decoded on the same held-out levels, so
their exchange rates are directly comparable. Output matches
``scripts/decode_h1.py``, one ``.npz`` per model.

One process rather than one per model, which is worth about a quarter of the
wall clock at this grid size: the demonstrations are loaded once, their
observations built once, and ``fast_decode``'s jitted step compiled once. Every
model has the same shape, so nothing recompiles as the parameters change.

The exchange-rate fit is skipped (``indifference=False``). It is four thousand
gradient steps and was 81% of the cost of a decode, and none of the analyses use
it -- they read the crossing off binned rates, because the fitted one is biased
by saturation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
from goalmisgen.offline.decode import replay_all
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.fast_decode import greedy_decode_cached
from goalmisgen.offline.train import list_checkpoints, load_checkpoint
from goalmisgen.volume import offset_tag

OFFSETS = (-0.45, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45)
"""Thirteen of the grid's twenty-five, evenly spread in the threshold they reach.

Eight of the design's twelve magnitudes sit between 0.38 and 0.45 and land within
a couple of steps of each other; that clustering buys leverage for *fitting* the
axis in weight space, which still uses every arm, and buys nothing for a curve
of threshold against offset. Two are kept at each extreme so the spread between
near-identical arms is visible.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("/workspace/data/offline/runs"))
    parser.add_argument("--demos", default="/workspace/data/offline/demos/test.rho100")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sweeps", nargs="+", default=["o0", "o1"])
    parser.add_argument("--steps", type=int, default=1000, help="Arm budget, part of the arm directory name.")
    parser.add_argument("--n", type=int, default=20_000, help="Levels per model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demos = DemoSet.load(args.demos, hide_values=True)
    indices = np.arange(min(args.n, len(demos)))
    observations = demos.observations(indices)

    values = np.asarray(demos.values)[indices]
    distances = np.asarray(demos.distances)[indices].astype(int)
    feature_ids = np.asarray(demos.feature_ids)[indices].astype(int)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(indices))
    # Fixed for every model in the grid, so built once.
    level = dict(
        d_rich=distances[rows, richer],
        d_poor=distances[rows, 1 - richer],
        colour_of_rich=feature_ids[rows, richer],
    )
    print(f"{len(indices):,} levels from {args.demos}")

    def decode_to(path: Path, model, params, what: str) -> None:
        if path.exists():
            print(f"  have {what}")
            return
        start = time.perf_counter()
        decoded = greedy_decode_cached(model, params, observations)
        outcomes = replay_all(demos, indices, decoded)
        reached = np.array([bool(o.get("reached_objective")) for o in outcomes])
        reached_fid = np.array([-1 if o.get("reached_feature_id") is None else int(o["reached_feature_id"]) for o in outcomes])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, reached=reached, reached_fid=reached_fid, **level)
        took = (reached_fid == level["colour_of_rich"])[reached].mean()
        print(f"  {what:<22} {time.perf_counter() - start:5.1f}s  reached {reached.mean():.3f}  took richer {took:.3f}")

    for seed in args.seeds:
        run = args.runs / f"bcnv11.s{seed}"
        base = load_base(run)
        decode_to(args.out / f"base.s{seed}.npz", base.model, base.params, f"base s{seed}")

        for sweep in args.sweeps:
            arms = arm_dirs(run, sweep, args.steps)
            print(f"seed {seed} {sweep}: {len(arms)} finished arms")
            for offset in OFFSETS:
                tag = f"{sweep}{offset_tag(offset)}"
                directory = arms.get(round(offset, 10))
                if directory is None:
                    print(f"  missing arm {tag}")
                    continue
                _, params = load_checkpoint(list_checkpoints(directory)[-1][1])
                decode_to(args.out / f"arms.s{seed}" / f"{tag}.npz", base.model, params, f"arm {tag}")

            # The axis uses every arm of the sweep; only the decodes are trimmed.
            diffs = load_diffs(base, arms)
            fitted = np.array(sorted(o for o in diffs if abs(o) > 1e-9))
            axis, _ = fit_axis_and_drift(fitted, np.stack([diffs[o] for o in fitted]))
            print(f"  axis from {len(fitted)} arms, |axis| {np.linalg.norm(axis):.4g}")
            for offset in OFFSETS:
                tag = f"{sweep}{offset_tag(offset)}"
                params = base.unravel(np.asarray(base.flat + offset * axis, dtype=np.float32))
                decode_to(args.out / f"written.s{seed}" / f"{tag}.npz", base.model, params, f"write {tag}")

    print("GRID_DONE")


if __name__ == "__main__":
    main()
