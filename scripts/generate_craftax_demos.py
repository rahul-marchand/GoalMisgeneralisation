"""Demonstrate a split of an ore-field dataset as Craftax routes.

    uv run python scripts/generate_craftax_demos.py --levels /workspace/data/levels/craftax/1.00-0.50@300k \
        --split train --rho 1.0 --out /workspace/data/craftax/demos/train.rho100

The twin of ``generate_demos.py``: same expert, same colour draws, so a set
here is paired level for level with a maze set at the same rho on the same
dataset. The routes are the maze routes in the engine's vocabulary with a DO
appended (:mod:`goalmisgen.craftax.demos`), and the dataset is verified
against the ore-field fingerprint rebuilt from its ``sampler.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from goalmisgen.craftax.demos import CraftaxDemoSet, CraftaxTask
from goalmisgen.craftax.levels import ore_field_fingerprint
from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.offline.demos import DEFAULT_MAX_ACTIONS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_levels import usable_cpus  # noqa: E402
from generate_ore_fields import load_sampler  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=Path, required=True, help="Ore-field dataset directory.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "valid", "test"))
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--colour-keyed", action="store_true", help="Pin kind i to objective i; see generate_demos.py.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=None, help="Only the first N levels of the split.")
    parser.add_argument("--workers", type=int, default=usable_cpus())
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--step-limit", type=int, default=120, help="The engine's limit; the expert gets one fewer.")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--no-verify", action="store_true", help="Skip the fingerprint check. Leave it on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = None if args.no_verify else ore_field_fingerprint(load_sampler(args.levels))
    dataset = LevelDataset.load(args.levels, expected_fingerprint=expected)
    if args.split not in dataset.stored_splits:
        raise SystemExit(f"dataset has no stored split {args.split!r}; it has {sorted(dataset.stored_splits)}")
    indices = np.asarray(dataset.stored_splits[args.split])
    if args.n is not None:
        indices = indices[: args.n]
    task = CraftaxTask()

    print(f"source      {args.levels}  ({dataset.fingerprint})")
    print(f"split       {args.split}: {len(indices):,} levels")
    print(f"rho         {args.rho}")
    print(f"task        kinds {task.kinds}, tools {task.tools}")
    print(f"workers     {args.workers}")
    print(f"output      {args.out}\n")

    start = time.perf_counter()
    demos = CraftaxDemoSet.generate(
        dataset,
        indices,
        rho=args.rho,
        task=task,
        colour_keyed=args.colour_keyed,
        seed=args.seed,
        step_penalty=args.step_penalty,
        step_limit=args.step_limit,
        max_actions=args.max_actions,
        workers=args.workers,
        split=args.split,
    )
    demos.save(args.out)
    elapsed = time.perf_counter() - start

    richer = np.argmax(demos.values, 1)
    print(f"routes      mean {demos.lengths.mean():.1f} actions, max {demos.lengths.max()} (DO included)")
    print(f"ambiguous   {demos.ambiguous.mean():.2%}")
    print(f"kind 0 on richer objective: {(demos.feature_ids[np.arange(len(demos)), richer] == 0).mean():.1%}")
    print(f"done in {elapsed:.1f}s ({1000 * elapsed / len(demos):.2f} ms/level wall clock)")


if __name__ == "__main__":
    main()
