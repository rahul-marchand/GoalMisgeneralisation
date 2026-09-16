"""One DAgger pass: roll a crafting model out, label what it visited, save the states.

    uv run python scripts/craft_dagger.py RUN_DIR DEMOS --fields 5000 --out /workspace/data/craftax/craft/demos-rh/dagger.s1 [--guard] [--states-per-route 8]

Then continue training with the pool and these states together:

    uv run python experiments/023_train_bc.py --demos <pool>/train.rho100 --extra-demos <out> --init-from <checkpoint> ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from goalmisgen.craftax import dagger
from goalmisgen.offline.demonstrations import load_demonstrations
from goalmisgen.offline.train import list_checkpoints, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("demos", type=Path, help="A receding crafting set; its first --fields fields are rolled out.")
    parser.add_argument("--fields", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0, help="First field of the range rolled out.")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--guard", action="store_true", help="Roll out with the ineffective-repeat guard.")
    parser.add_argument("--states-per-route", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.checkpoint is None else next(c for c in checkpoints if c[0] == args.checkpoint)
    model, params = load_checkpoint(directory)
    demos = load_demonstrations(args.demos, hide_values=True)
    indices = np.arange(args.start, min(args.start + args.fields, len(demos)))
    t = time.perf_counter()
    states = dagger.collect(
        model,
        params,
        demos,
        indices,
        seed=args.seed,
        avoid_ineffective_repeat=args.guard,
        states_per_route=args.states_per_route,
    )
    states.save(args.out)
    lengths = np.asarray(states.lengths)
    inv = np.asarray(states.inventory)
    print(
        f"{args.run.name} @ step {step}: {len(indices)} fields rolled out, {len(states)} states labelled in {time.perf_counter() - t:.0f}s"
    )
    print(
        f"labels: mean remaining route {lengths.mean():.1f} actions; target iron on {np.mean(np.asarray(states.feature_ids)[np.arange(len(states)), np.asarray(states.target)] == 0):.0%}"
    )
    print(
        f"states: wood pickaxe held {np.mean(inv[:, 2] > 0):.0%}, stone pickaxe {np.mean(inv[:, 3] > 0):.0%}, stone in hand {np.mean(inv[:, 1] > 0):.0%}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
