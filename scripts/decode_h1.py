"""Decode one route model on the held-out set, keeping both distances per level.

    uv run python scripts/decode_h1.py /workspace/data/offline/runs/bcnv11.s1 50000 out.npz

Writes one row per level: the distance to each objective, which colour marks
the richer one, and which objective the model actually reached. That is enough
to ask whether the choice is a function of the two distances alone -- group
levels by the exact pair ``(d_rich, d_poor)`` and see whether a cell is
unanimous -- without re-running the model.

Values are hidden, matching how ``bcnv11`` was trained: the observation carries
colour but not what a colour is worth, so the exchange rate is a learned
constant rather than an input.

Distances come from the dataset rather than the decode. They route around the
other objective (:func:`goalmisgen.envs.solver.objective_distances`), because
reaching either objective ends the episode and a route that crosses the other
one never arrives.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.fast_decode import greedy_decode_cached
from goalmisgen.offline.train import list_checkpoints, load_checkpoint

DEFAULT_DEMOS = "/workspace/data/offline/demos/test.rho100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="A route-model run directory; its last checkpoint is used.")
    parser.add_argument("n", type=int, help="How many levels of the split to decode.")
    parser.add_argument("out", type=Path, help="Where to write the per-level arrays (.npz).")
    parser.add_argument("--demos", type=str, default=DEFAULT_DEMOS, help="Demonstration set to decode against.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demos = DemoSet.load(args.demos, hide_values=True)
    model, params = load_checkpoint(list_checkpoints(args.run)[-1][1])
    indices = np.arange(min(args.n, len(demos)))

    start = time.perf_counter()
    summary, _, outcomes = evaluate(model, params, demos, indices, decoder=greedy_decode_cached, indifference=False)
    print(f"decoded {len(indices):,} in {time.perf_counter() - start:.1f}s")
    print(summary)

    values = np.asarray(demos.values)[indices]
    distances = np.asarray(demos.distances)[indices].astype(int)
    feature_ids = np.asarray(demos.feature_ids)[indices].astype(int)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(indices))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        d_rich=distances[rows, richer],
        d_poor=distances[rows, 1 - richer],
        colour_of_rich=feature_ids[rows, richer],
        reached=np.array([bool(o.get("reached_objective")) for o in outcomes]),
        # -1 where nothing was reached, so the column stays integral.
        reached_fid=np.array([-1 if o.get("reached_feature_id") is None else int(o["reached_feature_id"]) for o in outcomes]),
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
