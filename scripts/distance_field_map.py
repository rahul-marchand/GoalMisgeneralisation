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

from goalmisgen.analysis import collect_rollouts, fields, targets
from goalmisgen.analysis.probes import Feature
from goalmisgen.configs.env import MazeConfig

N_FEATURES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--examples", type=int, default=12, help="Mazes drawn in full.")
    parser.add_argument("--score-episodes", type=int, default=512, help="Episodes behind the accuracy breakdown.")
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

    activations = Feature("activations", operator.attrgetter("features"))
    target = targets.DistanceToObjective(targets.fixed(args.feature), name=f"d->f{args.feature}", n_features=N_FEATURES)

    train = fields.cell_data(rollouts(0, args.train_episodes), activations, target)

    # The whole scoring set, so accuracy can be read in cells and broken down by
    # distance and by detour size. A handful of drawn mazes shows the shape of
    # the error; only the full set says how big it is.
    scoring = rollouts(9999, args.score_episodes)
    scored = fields.cell_data(scoring, activations, target)
    _, all_predictions, l2, _ = fields.fit_predict(train, scored)
    print(f"fitted on {len(train.y):,} cells, l2={l2:g}; scored on {len(scored.y):,}")

    examples = scoring[: args.examples]
    test = fields.cell_data(examples, activations, target)
    _, prediction, _, _ = fields.fit_predict(train, test)

    truth, predicted, straight, observations = [], [], [], []
    cursor = 0
    for rollout in examples:
        labels = target.labels(rollout)
        confound = target.confound(rollout)
        usable = np.isfinite(labels) & np.isfinite(confound).all(axis=-1)

        field = np.full(labels.shape, np.nan)
        field[usable] = prediction[cursor : cursor + int(usable.sum())]
        cursor += int(usable.sum())

        truth.append(np.where(usable, labels, np.nan))
        predicted.append(field)
        straight.append(np.where(usable, confound[:, :, 0], np.nan))
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
        # Flat arrays over every scored cell, for the accuracy breakdown.
        all_true=scored.y,
        all_predicted=all_predictions,
        all_straight=scored.confound[:, 0],
        all_episode=scored.episode,
    )
    print(f"wrote {args.out}  ({len(truth)} episodes)")


if __name__ == "__main__":
    main()
