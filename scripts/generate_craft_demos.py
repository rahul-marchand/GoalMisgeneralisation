"""Demonstrate a range of the crafting pool at one correlation and one pair of values.

    uv run python scripts/generate_craft_demos.py --seed 0 --start 0 --count 100000 --rho 1.0 \\
        --out /workspace/data/craftax/craft/demos/train.rho100

There is no separate level dataset for the crafting task: field ``i`` of the
pool ``(sampler, seed)`` is a deterministic draw, so a set is addressed by an
index range, splits are disjoint ranges, and ``shared_levels`` compares sets
from one pool by index. Values are fixed and consume no randomness, so sets
at different values share layouts and a value sweep is paired field for field.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from goalmisgen.craftax.craft import CraftDemoSet, CraftTask, FieldSampler
from goalmisgen.envs.values import FixedValues

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_levels import usable_cpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="The pool.")
    parser.add_argument("--start", type=int, required=True, help="First field index of the range.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--colour-seed", type=int, default=0)
    parser.add_argument("--objective-values", type=float, nargs="+", default=(2.0, 0.5))
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=6)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=usable_cpus())
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--max-actions", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampler = FieldSampler(
        size=args.size, obstacle_density=args.density, n_trees=args.trees, values=FixedValues(tuple(args.objective_values))
    )
    task = CraftTask()
    print(f"pool        seed {args.seed}, fields {args.start:,}..{args.start + args.count:,}")
    print(f"sampler     {sampler}")
    print(f"task        kinds {task.kinds}")
    print(f"rho         {args.rho}")
    print(f"workers     {args.workers}")
    print(f"output      {args.out}\n")
    start = time.perf_counter()
    demos = CraftDemoSet.generate(
        sampler,
        seed=args.seed,
        start=args.start,
        count=args.count,
        rho=args.rho,
        task=task,
        colour_seed=args.colour_seed,
        step_penalty=args.step_penalty,
        step_limit=args.step_limit,
        max_actions=args.max_actions,
        workers=args.workers,
        split=args.split,
    )
    demos.save(args.out)
    elapsed = time.perf_counter() - start
    rows = np.arange(len(demos))
    richer = np.argmax(demos.values, 1)
    took_richer = demos.target == richer
    print(f"routes      mean {demos.lengths.mean():.1f} actions, max {demos.lengths.max()}")
    print(f"took richer {took_richer.mean():.1%}   ambiguous {demos.ambiguous.mean():.2%}")
    extra = demos.distances[rows, richer] - demos.distances[rows, 1 - richer]
    print(f"extra cost of the richer chain p5/50/95 = {np.percentile(extra, [5, 50, 95]).round().tolist()}")
    print(f"done in {elapsed:.1f}s ({1000 * elapsed / len(demos):.1f} ms/field wall clock)")


if __name__ == "__main__":
    main()
