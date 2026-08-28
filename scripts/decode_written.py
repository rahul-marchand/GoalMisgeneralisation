"""Decode models written from the value axis, rather than fine-tuned.

    uv run python scripts/decode_written.py RUN --sweep o0 --offsets 0.45 0.30 ... --out DIR

An arm is a network that was trained; a *written* model is the base plus
``offset * axis``, a pure linear intervention. ``027`` shows the two differ on
the mean (leave-one-out write error 2.62 steps), so a shape measured on arms
does not carry over to writes and both have to be decoded.

The axis is fitted from **every** arm of the sweep even when only some offsets
are written: fitting a direction in weight space is what the wide grid's
leverage is for, and it costs no decode. ``--drift`` adds the common fine-tuning
component, which is the better-calibrated write in ``027`` (1.30 steps against
2.62) and the one whose exchange rates line up with the arms'.

Output matches ``scripts/decode_h1.py`` so the same analysis reads both.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.fast_decode import greedy_decode_cached
from goalmisgen.volume import offset_tag

DEFAULT_DEMOS = "/workspace/data/offline/demos/test.rho100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Base run directory, e.g. .../runs/bcnv11.s1")
    parser.add_argument("--sweep", default="o0")
    parser.add_argument("--steps", type=int, default=1000, help="Arm budget, part of the arm directory name.")
    parser.add_argument("--offsets", type=float, nargs="+", required=True, help="Which offsets to write and decode.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--demos", default=DEFAULT_DEMOS)
    parser.add_argument("--drift", action="store_true", help="Add the common fine-tuning component to the write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_base(args.run)
    arms = arm_dirs(args.run, args.sweep, args.steps)
    if len(arms) < 4:
        raise SystemExit(f"{args.run} has {len(arms)} finished {args.sweep} arms; need at least 4 to fit an axis")
    diffs = load_diffs(base, arms)
    offsets = np.array(sorted(o for o in diffs if abs(o) > 1e-9))
    axis, drift = fit_axis_and_drift(offsets, np.stack([diffs[o] for o in offsets]))
    print(f"axis fitted from {len(offsets)} arms; |axis| {np.linalg.norm(axis):.4g} |drift| {np.linalg.norm(drift):.4g}")

    demos = DemoSet.load(args.demos, hide_values=True)
    indices = np.arange(min(args.n, len(demos)))
    values = np.asarray(demos.values)[indices]
    distances = np.asarray(demos.distances)[indices].astype(int)
    feature_ids = np.asarray(demos.feature_ids)[indices].astype(int)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(indices))
    args.out.mkdir(parents=True, exist_ok=True)

    for offset in args.offsets:
        tag = f"{args.sweep}{offset_tag(offset)}"
        path = args.out / f"{tag}.npz"
        if path.exists():
            print(f"have {tag}")
            continue
        flat = base.flat + offset * axis + (drift if args.drift else 0.0)
        params = base.unravel(np.asarray(flat, dtype=np.float32))
        start = time.perf_counter()
        summary, _, outcomes = evaluate(base.model, params, demos, indices, decoder=greedy_decode_cached, indifference=False)
        print(f"{tag}: {time.perf_counter() - start:.0f}s  {summary}")
        np.savez(
            path,
            d_rich=distances[rows, richer],
            d_poor=distances[rows, 1 - richer],
            colour_of_rich=feature_ids[rows, richer],
            reached=np.array([bool(o.get("reached_objective")) for o in outcomes]),
            reached_fid=np.array(
                [-1 if o.get("reached_feature_id") is None else int(o["reached_feature_id"]) for o in outcomes]
            ),
        )


if __name__ == "__main__":
    main()
