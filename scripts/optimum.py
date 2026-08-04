"""Mean optimal utility of the levels an evaluation arm actually scores.

    uv run python scripts/optimum.py --levels /workspace/data/levels11rv --randomise-values

An evaluation curve only means something against the best score achievable on
the levels it was scored on. cleanba's evaluator re-creates its environments
with a fixed seed and resets ``n_episode_multiple`` times, so every evaluation
scores the *same* small batch — 128 levels at our settings — for the whole run.

Averaging over the entire split instead answers a different question, and on the
run this was written for it put the reference far too high: the batch mean is
2.7 standard errors below the split mean, so an agent at 96% of what was
achievable looked like it was at 59%. This prints both, so the gap is visible
rather than assumed away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import EVAL_SEED_OFFSET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=str, default=None, help="Level dataset. Omit to sample live.")
    parser.add_argument("--split", type=str, default="valid", help="What the evaluation arms draw from.")
    parser.add_argument("--min-size", type=int, default=11)
    parser.add_argument("--max-size", type=int, default=11)
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument("--seed", type=int, default=1234, help="The run's training seed; the offset is added here.")
    parser.add_argument("--num-envs", type=int, default=64, help="Must match the run's eval_num_envs.")
    parser.add_argument("--episode-multiple", type=int, default=2, help="Must match the run's n_episode_multiple.")
    parser.add_argument("--reference-resets", type=int, default=128, help="Resets used for the whole-split reference.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def optimal_utilities(config: MazeConfig, resets: int, step_penalty: float) -> np.ndarray:
    """Utility of the best objective on each level of the first ``resets`` batches.

    Repeated resets are how cleanba reaches the Nth batch of levels, so this
    walks the same sequence. Which levels come out does not depend on the
    policy, only on how many resets have happened.
    """
    envs = config.make()
    try:
        utilities = []
        for _ in range(resets):
            _, info = envs.reset()
            utilities.append(np.asarray(info["optimal_value"]) - step_penalty * np.asarray(info["optimal_distance"]))
        return np.concatenate(utilities)
    finally:
        envs.close()


def main() -> None:
    args = parse_args()

    def config(seed: int) -> MazeConfig:
        return MazeConfig(
            max_episode_steps=args.max_episode_steps,
            num_envs=args.num_envs,
            min_size=args.min_size,
            max_size=args.max_size,
            step_penalty=args.step_penalty,
            randomise_values=args.randomise_values,
            level_dataset=args.levels,
            **({"dataset_split": args.split} if args.levels else {}),
            asynchronous=False,
            seed=seed,
        )

    batch = optimal_utilities(config(args.seed + EVAL_SEED_OFFSET), args.episode_multiple, args.step_penalty)
    # A different seed, so the reference is the split rather than a larger
    # sample of the same corner of it.
    reference = optimal_utilities(config(args.seed + 2 * EVAL_SEED_OFFSET), args.reference_resets, args.step_penalty)

    sem = float(reference.std(ddof=1) / np.sqrt(len(batch)))
    result = {
        "eval_batch": {"mean": float(batch.mean()), "n": int(batch.size)},
        "split": {"mean": float(reference.mean()), "sd": float(reference.std(ddof=1)), "n": int(reference.size)},
        "batch_offset_in_sem": (float(batch.mean()) - float(reference.mean())) / sem,
        "levels": args.levels or "sampled live",
        "split_name": args.split,
        "randomise_values": args.randomise_values,
    }

    drawn_from = f"{args.split} split" if args.levels else "live sampling"
    print(f"eval batch  {result['eval_batch']['mean']:.3f}  over {result['eval_batch']['n']} levels")
    print(f"reference   {result['split']['mean']:.3f}  over {result['split']['n']} levels from the same {drawn_from}")
    print(f"\nthe batch sits {result['batch_offset_in_sem']:+.1f} sem from the reference mean")
    print("Use the batch figure as the reference line; it is what the curves were scored against.")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
