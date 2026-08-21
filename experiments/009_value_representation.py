"""Has an objective's value travelled to where the decision is made?

    uv run python experiments/009_value_representation.py CHECKPOINT --levels DIR --randomise-values

Value is an episode-level scalar, not a field. "Objective 0 is worth 0.72" is
the same number at every cell, so the within-episode stratification that carries
the distance results is zero here by construction. This measures it the shape it
actually is: one activation vector per episode, one label per episode.

And the interesting question is not whether value is decodable — the encoder
writes it into a channel, so at the objective's own cell it is there for free.
It is **where** it is decodable:

``objective cell``   trivial, and reported only as a reference. A 1x1 probe on
                     the observation gets this one too.
``agent cell``       the state at the cell the agent occupies, which is where a
                     comparison would have to happen for the policy to act on it.
``pooled``           averaged over every free cell.

Away from the objective the observation's value channel is zero, so anything the
activations give there is something the network moved. That is the prerequisite
for value being compared with distance at all, and it is what makes a steering
intervention on value meaningful rather than a roundabout way of editing the
input.

Needs an agent trained on **randomised** values: with fixed values there is
nothing to correlate against.
"""

from __future__ import annotations

import argparse
import operator
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen import provenance
from goalmisgen.analysis import collect_rollouts, geometry, metrics
from goalmisgen.analysis.probes import Feature, apply_linear, fit_ridge, layer_slice
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

        xs.append(vector)
        ys.append(geometry.objective_value(observation, feature_id, N_FEATURES))

    return np.stack(xs).astype(np.float64), np.array(ys, dtype=np.float64)


def score(train, test, seed: int = 0):
    """Correlation between predicted and true value, with an episode bootstrap."""
    (x_train, y_train), (x_test, y_test) = train, test
    if y_train.std() < 1e-9:
        raise RuntimeError("the objective's value does not vary; this needs an agent trained on randomised values")

    weights, mean, std = fit_ridge(x_train, y_train, l2=1.0)
    predicted = apply_linear(x_test, weights, mean, std)

    episodes = np.arange(len(y_test))
    correlation = float(np.corrcoef(predicted, y_test)[0, 1])
    low, high = metrics.bootstrap_episodes(
        lambda rows: float(np.corrcoef(predicted[rows], y_test[rows])[0, 1]) if len(set(rows.tolist())) > 2 else np.nan,
        episodes,
        seed=seed,
    )
    return correlation, (low, high), float(np.mean(np.abs(predicted - y_test)))


def main() -> None:
    args = parse_args()
    print(provenance.header())

    def env_config(seed: int, split: str) -> MazeConfig:
        settings: dict[str, object] = dict(
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

    values = np.array([geometry.objective_value(r.observation, args.feature, N_FEATURES) for r in test_trained])
    print(f"\nfeature {args.feature} is worth {values.min():.2f} to {values.max():.2f} across {len(values)} episodes")
    print(f"reading {'the last layer only' if args.last_layer_only else 'all layers'}\n")

    print(f"{'site':>16}{'arm':>14}{'correlation':>13}{'95% CI':>18}{'MAE':>8}")
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
            print(f"{site:>16}{name:>14}{correlation:>13.3f}{f'[{low:.3f}, {high:.3f}]':>18}{mae:>8.3f}")

        # Shuffling the labels breaks the pairing while keeping their
        # distribution, so anything it still finds is leakage in the fit.
        x_fit, y_fit = episode_rows(fit_trained, activations, args.feature, site)
        x_test, y_test = episode_rows(test_trained, activations, args.feature, site)
        shuffled = np.random.default_rng(args.seed).permutation(y_fit)
        correlation, (low, high), mae = score((x_fit, shuffled), (x_test, y_test), seed=args.seed)
        print(f"{site:>16}{'shuffled':>14}{correlation:>13.3f}{f'[{low:.3f}, {high:.3f}]':>18}{mae:>8.3f}\n")

    print(
        "The objective cell is a reference, not a result: the encoder writes the value there,\n"
        "so the observation arm gets it too. The agent cell is the one that matters — value has\n"
        "to reach where the policy acts before it can be compared with anything."
    )


if __name__ == "__main__":
    main()
