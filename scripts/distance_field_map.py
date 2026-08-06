"""Dump the probe's distance field beside the true one, so it can be drawn.

    uv run python scripts/distance_field_map.py CHECKPOINT --levels DIR --out fields.npz

The pilot reports a partial correlation of 0.71 but a detour R² of only 0.23,
which together say the probe orders cells correctly while under-predicting
magnitude where walls force a detour. That is a claim about *shape*, and a table
cannot show it. This saves the fitted probe's prediction at every free cell for a
handful of held-out episodes, so the error can be looked at directly.
"""

from __future__ import annotations

import argparse
import operator
from pathlib import Path

import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import collect_rollouts, fields, geometry
from goalmisgen.analysis.probes import apply_linear, fit_ridge
from goalmisgen.configs.env import MazeConfig

N_FEATURES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--feature", type=int, default=0)
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

    activations = operator.attrgetter("features")
    train = fields.field_dataset(rollouts(0, args.train_episodes), activations, args.feature, N_FEATURES)
    l2 = fields.choose_l2(train)
    weights, mean, std = fit_ridge(train.x, train.y, l2=l2)
    print(f"fitted on {len(train.y):,} cells, l2={l2:g}")

    truth, predicted, straight, observations = [], [], [], []
    for rollout in rollouts(9999, args.examples):
        labels, manhattan, _ = fields.distance_target(rollout, args.feature, N_FEATURES)
        usable = geometry.free_cells(rollout.observation) & np.isfinite(labels) & (labels > 0)

        field = np.full(labels.shape, np.nan)
        field[usable] = apply_linear(rollout.features[usable].astype(np.float64), weights, mean, std)

        truth.append(np.where(usable, labels, np.nan))
        predicted.append(field)
        straight.append(np.where(usable, manhattan, np.nan))
        observations.append(rollout.observation)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        truth=np.stack(truth),
        predicted=np.stack(predicted),
        straight=np.stack(straight),
        observations=np.stack(observations),
        feature=args.feature,
        checkpoint=str(args.checkpoint),
    )
    print(f"wrote {args.out}  ({len(truth)} episodes)")


if __name__ == "__main__":
    main()
