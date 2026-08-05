"""Dump per-cell probe scores for a few episodes, so the plan can be drawn.

    uv run python scripts/plan_heatmap.py CHECKPOINT --levels DIR --out plan_examples.npz

The AUC table says the route is decodable. It does not let anyone *see* it.
This fits the same probe experiments/003_probe_plan.py fits, then saves, for a
handful of held-out episodes, the score the probe gives every free cell
alongside the route the agent actually walked.

Runs where the GPU is: the arrays it writes are small and plot fine anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import cell_dataset, collect_rollouts
from goalmisgen.analysis.probes import apply_logistic, fit_logistic
from goalmisgen.configs.env import MazeConfig
from goalmisgen.envs.observation import WALL_CHANNEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--examples", type=int, default=12, help="Episodes to save. Draw the clearest few.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def env_config(seed: int) -> MazeConfig:
        settings = dict(
            max_episode_steps=120,
            num_envs=args.num_envs,
            min_size=args.size,
            max_size=args.size,
            feature_value_correlation=args.correlation,
            randomise_values=args.randomise_values,
            level_dataset=args.levels,
            asynchronous=False,
            seed=seed,
        )
        if args.levels:
            settings["dataset_split"] = args.split
        return MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=env_config(0))
    print(f"checkpoint {args.checkpoint.name}  (update {update})")

    def rollouts(seed: int, episodes: int):
        return collect_rollouts(env_config(seed).make(), policy, train_state.params, episodes, seed=seed)

    # Fitted on one set of levels, drawn on another, exactly as the AUC is scored.
    train = rollouts(0, args.train_episodes)
    weights, mean, std = fit_logistic(*cell_dataset(train))
    examples = rollouts(9999, args.examples)

    scores, visited, visit_step, observations = [], [], [], []
    for rollout in examples:
        free = rollout.observation[:, :, WALL_CHANNEL] < 0.5
        grid = np.full(free.shape, np.nan)
        grid[free] = apply_logistic(rollout.features[free].astype(np.float64), weights, mean, std)
        scores.append(grid)
        visited.append(rollout.visited)
        visit_step.append(rollout.visit_step)
        observations.append(rollout.observation)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        scores=np.stack(scores),
        visited=np.stack(visited),
        visit_step=np.stack(visit_step),
        observations=np.stack(observations),
        checkpoint=str(args.checkpoint),
    )
    print(f"wrote {args.out}  ({len(scores)} episodes)")


if __name__ == "__main__":
    main()
