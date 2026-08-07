"""Is the objective the agent will take already decided before it moves?

    uv run python experiments/011_target_probe.py CHECKPOINT --levels DIR

The founding hypothesis of the project is that an agent's internal
representations predict what it will do before its behaviour shows it. Every
probe so far has read a *quantity* the agent might use — a route, a distance
field, a value. This reads the **decision**: from the recurrent state at t=0,
before a single action, which of the two objectives will it end up at.

Read as classification, one episode per row, at three sites.

``objective cells``  the two candidates' own cells
``agent cell``       where the policy acts
``pooled``           averaged over free cells

The observation arm is the control that matters and it is not trivially weak:
which objective *should* be taken is fully determined by the level, so anything
that could compute utility from the input would predict the choice. But a 1x1
probe sees one cell's channels and cannot compute distances, so it should be
near chance — and the gap to the activation probe is how much the network has
already worked out.

On an agent trained without a value channel this is the only well-posed probe
of its goal representation. Its values are learned constants rather than
per-episode variables, so there is no value to correlate against; the decision,
however, varies every episode.
"""

from __future__ import annotations

import argparse
import operator
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import collect_rollouts, geometry, metrics
from goalmisgen.analysis.probes import Feature, apply_logistic, fit_logistic, layer_slice
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33

N_FEATURES = 2
N_LAYERS = 3
LAST_LAYER = N_LAYERS - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--fit-split", type=str, default="valid")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit-episodes", type=int, default=1024)
    parser.add_argument("--test-episodes", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=0.5)
    parser.add_argument("--feature", type=int, default=0)
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument(
        "--hide-values",
        action="store_true",
        help="Match a run trained without a value channel; its observations have one channel fewer.",
    )
    parser.add_argument("--last-layer-only", action="store_true", help="What the actor reads.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def episode_rows(rollouts, feature: Feature, feature_id: int, site: str):
    """One activation vector and one value per episode, read at ``site``."""
    xs, ys = [], []
    for rollout in rollouts:
        observation = rollout.observation
        geometry.check_layout(observation, N_FEATURES)
        grid = feature(rollout)

        if site == "objective cell":
            row, col = geometry.objective_cell(observation, feature_id)
            vector = grid[row, col]
        elif site == "agent cell":
            row, col = geometry.agent_cell(observation)
            vector = grid[row, col]
        elif site == "pooled":
            free = geometry.free_cells(observation)
            # The objective's own cell is excluded: the encoder wrote the value
            # there, so including it would let the average smuggle in the input.
            free[geometry.objective_cell(observation, feature_id)] = False
            vector = grid[free].mean(axis=0)
        else:
            raise ValueError(f"unknown site {site!r}")

        reached = rollout.info.get("reached_feature_id")
        if reached is None:
            continue  # no objective taken, so there is no decision to predict
        xs.append(vector)
        ys.append(float(reached == feature_id))

    return np.stack(xs).astype(np.float64), np.array(ys, dtype=np.float64)


def score(train, test, seed: int = 0):
    """AUC for predicting the objective taken, with an episode bootstrap."""
    (x_train, y_train), (x_test, y_test) = train, test
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("the agent took the same objective in every episode; there is no decision to predict")

    weights, mean, std = fit_logistic(x_train, y_train, steps=800)
    predicted = apply_logistic(x_test, weights, mean, std)

    episodes = np.arange(len(y_test))
    auc = metrics.roc_auc(y_test, predicted)
    low, high = metrics.bootstrap_episodes(
        lambda rows: metrics.roc_auc(y_test[rows], predicted[rows]), episodes, seed=seed
    )
    accuracy = float(((predicted >= 0.5) == y_test).mean())
    return auc, (low, high), accuracy


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}")

    def env_config(seed: int, split: str) -> MazeConfig:
        settings: dict[str, object] = dict(
            max_episode_steps=120,
            num_envs=args.num_envs,
            min_size=args.size,
            max_size=args.size,
            feature_value_correlation=args.correlation,
            randomise_values=args.randomise_values,
            value_encoding="none" if args.hide_values else "at_objective",
            colour_is_the_only_value_cue=args.hide_values,
            level_dataset=args.levels,
            asynchronous=False,
            seed=seed,
        )
        if args.levels:
            settings["dataset_split"] = split
        return MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=env_config(0, args.fit_split))
    print(f"checkpoint {args.checkpoint.name}  (update {update})")

    untrained = maze_drc33(min_size=args.size, max_size=args.size).net.init_params(
        env_config(0, args.fit_split).make(), jax.random.PRNGKey(12345)
    )[2]

    whole = Feature("activations", operator.attrgetter("features"))
    activations = layer_slice(whole, LAST_LAYER, N_LAYERS) if args.last_layer_only else whole
    observation = Feature("observation", operator.attrgetter("observation"))

    def rollouts(seed, episodes, split, probe_params=None):
        return collect_rollouts(
            env_config(seed, split).make(),
            policy,
            train_state.params,
            episodes,
            seed=seed,
            probe_params=probe_params,
        )

    fit_trained = rollouts(0, args.fit_episodes, args.fit_split)
    test_trained = rollouts(9999, args.test_episodes, args.split)
    fit_untrained = rollouts(0, args.fit_episodes, args.fit_split, untrained)
    test_untrained = rollouts(9999, args.test_episodes, args.split, untrained)

    taken = np.array([r.info.get("reached_feature_id") == args.feature for r in test_trained if r.info.get("reached_feature_id") is not None])
    print(f"\nfeature {args.feature} was taken in {taken.mean():.1%} of {len(taken)} decided episodes")
    print(f"reading {'the last layer only' if args.last_layer_only else 'all layers'}\n")

    print(f"{'site':>16}{'arm':>14}{'AUC':>13}{'95% CI':>18}{'accuracy':>10}")
    for site in ("objective cell", "agent cell", "pooled"):
        arms = [
            ("trained", activations, fit_trained, test_trained),
            ("untrained", activations, fit_untrained, test_untrained),
            ("observation", observation, fit_trained, test_trained),
        ]
        for name, feature, fit_set, test_set in arms:
            correlation, (low, high), mae = score(
                episode_rows(fit_set, feature, args.feature, site),
                episode_rows(test_set, feature, args.feature, site),
                seed=args.seed,
            )
            print(f"{site:>16}{name:>14}{correlation:>13.3f}{f'[{low:.3f}, {high:.3f}]':>18}{mae:>10.1%}")

        # Shuffling the labels breaks the pairing while keeping their
        # distribution, so anything it still finds is leakage in the fit.
        x_fit, y_fit = episode_rows(fit_trained, activations, args.feature, site)
        x_test, y_test = episode_rows(test_trained, activations, args.feature, site)
        shuffled = np.random.default_rng(args.seed).permutation(y_fit)
        correlation, (low, high), mae = score((x_fit, shuffled), (x_test, y_test), seed=args.seed)
        print(f"{site:>16}{'shuffled':>14}{correlation:>13.3f}{f'[{low:.3f}, {high:.3f}]':>18}{mae:>10.1%}\n")

    print(
        "A 1x1 probe on the observation cannot compute distances, so it cannot work out which\n"
        "objective is worth taking however fully the level determines it. The gap between that\n"
        "and the activation probe is how much the network has already decided at t=0."
    )


if __name__ == "__main__":
    main()
