"""Demonstrate a split of a level dataset at one correlation.

    uv run python scripts/generate_demos.py --levels /workspace/data/levels/values/1.00-0.50@1M \
        --split train --rho 1.0 --out /workspace/data/offline/demos/train.rho100

One demonstration per level: the BFS-optimal route under the task's utility,
with colours assigned at ``--rho``. The same split at several correlations
gives sets paired level for level - only the colours differ - which is what
makes a behavioural comparison across rho attributable to the colours alone.

About a millisecond per level, CPU-bound, embarrassingly parallel.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from goalmisgen.configs.env import MazeConfig
from goalmisgen.envs.dataset import LevelDataset, dataset_fingerprint
from goalmisgen.offline.demos import DEFAULT_MAX_ACTIONS, DemoSet

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_levels import usable_cpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=Path, required=True, help="Source level dataset directory.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "valid", "test"))
    parser.add_argument("--rho", type=float, required=True, help="Colour-value correlation to demonstrate at.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seeds the colour draws, per level.")
    parser.add_argument("--n", type=int, default=None, help="Demonstrate only the first N levels of the split.")
    parser.add_argument("--workers", type=int, default=usable_cpus())
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--step-limit", type=int, default=120)
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--min-size", type=int, default=11)
    parser.add_argument("--max-size", type=int, default=11)
    parser.add_argument("--objective-values", type=float, nargs="+", default=(1.0, 0.5))
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the fingerprint check against the source dataset. Leave it on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MazeConfig(
        max_episode_steps=args.step_limit,
        min_size=args.min_size,
        max_size=args.max_size,
        step_penalty=args.step_penalty,
        objective_values=tuple(args.objective_values),
    )
    expected = None if args.no_verify else dataset_fingerprint(config.live_sampler())
    dataset = LevelDataset.load(args.levels, expected_fingerprint=expected)
    if args.split not in dataset.stored_splits:
        raise SystemExit(f"dataset has no stored split {args.split!r}; it has {sorted(dataset.stored_splits)}")
    indices = np.asarray(dataset.stored_splits[args.split])
    if args.n is not None:
        indices = indices[: args.n]

    print(f"source      {args.levels}  ({dataset.fingerprint})")
    print(f"split       {args.split}: {len(indices):,} levels")
    print(f"rho         {args.rho}")
    print(f"workers     {args.workers}")
    print(f"output      {args.out}\n")

    start = time.perf_counter()
    demos = DemoSet.generate(
        dataset,
        indices,
        rho=args.rho,
        seed=args.seed,
        step_penalty=args.step_penalty,
        step_limit=args.step_limit,
        max_actions=args.max_actions,
        workers=args.workers,
        split=args.split,
    )
    demos.save(args.out)
    elapsed = time.perf_counter() - start

    size_mb = sum(f.stat().st_size for f in args.out.rglob("*")) / 1e6
    print(f"routes      mean {demos.lengths.mean():.1f} moves, max {demos.lengths.max()}")
    print(f"ambiguous   {demos.ambiguous.mean():.2%}")
    print(
        f"feature 0 on richer objective: {(demos.feature_ids[np.arange(len(demos)), np.argmax(demos.values, 1)] == 0).mean():.1%}"
    )
    print(f"done in {elapsed:.1f}s ({1000 * elapsed / len(demos):.2f} ms/level wall clock) -> {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
