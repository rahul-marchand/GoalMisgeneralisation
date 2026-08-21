"""Is the agent's route decodable from its hidden state before it moves?

    uv run python experiments/003_probe_plan.py CHECKPOINT --levels DIR --size 11

This is the planning-interpretability result reproduced in our environment. A
linear readout, applied identically at every maze cell, tries to predict which
cells the agent will step on — from the DRC's state at t=0, before it has taken
an action.

The comparison that carries the claim is against an **untrained network of the
same shape**, not against the raw observation. A pointwise linear readout of a
one-hot observation cannot express any spatial relation, so beating it may cost
nothing but having a receptive field: on synthetic labels a random convolutional
tower scores 0.76 against the observation's 0.60, and a feature holding only
distance-from-agent scores 0.80, neither containing a plan. So:

* trained well above untrained
  → training put the route there, which is the result
* trained ≈ untrained
  → the score is architecture, not learning, and there is nothing to report
* both near chance
  → no decodable plan, which would be the interesting negative result

``--think`` re-applies the network to the initial observation *for the probe
only*. Letting it change the agent's actions would change the labels too, so a
rising score would be unattributable: route quality alone moves this measure by
0.3 AUC.

Distance bands use **distance-matched negatives**: a cell reached at step k is
scored only against never-visited cells that are also k steps away. Pooled
negatives let a pure distance feature reproduce the entire "still high far out"
profile, which is otherwise read as evidence of a plan.

Intervals are bootstrapped over **episodes**, not cells. Cells within an episode
share a maze and a route, so the cell-level interval is roughly 1.6x too narrow.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import auc_interval, collect_rollouts, probe, probe_by_distance
from goalmisgen.analysis.probes import ProbeSource
from goalmisgen.configs.env import MazeConfig
from goalmisgen.nets.readers import state_reader_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--correlation", type=float, default=1.0, help="Correlation the probe is fitted at.")
    parser.add_argument(
        "--test-correlations",
        type=float,
        nargs="+",
        default=None,
        help="Correlations the fitted probe is scored at; defaults to the fitting one. Scoring "
        "at rho=0.0 a probe fitted at rho=1.0 asks whether the route is still readable "
        "once the proxy is reversed - the representational twin of the behavioural gap.",
    )
    parser.add_argument(
        "--think",
        type=int,
        nargs="+",
        default=[0],
        help="Extra forward passes on the initial observation before probing.",
    )
    parser.add_argument(
        "--by-distance",
        action="store_true",
        help="Also score the probe separately at each distance along the route.",
    )
    parser.add_argument(
        "--randomise-values",
        action="store_true",
        help="Must match the run being probed: the value scheme is part of a dataset's fingerprint.",
    )
    parser.add_argument(
        "--no-untrained",
        action="store_true",
        help="Skip the untrained-network control, which costs a second set of rollouts.",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Also score each layer's share of the state on its own. Which layer holds the "
        "route is what a cross-architecture comparison turns on: a DRC's actor reads only "
        "its last layer, a transformer's readout sees the final residual stream.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def env_config(seed: int, correlation: float | None = None) -> MazeConfig:
        settings = dict(
            max_episode_steps=120,
            num_envs=args.num_envs,
            min_size=args.size,
            max_size=args.size,
            feature_value_correlation=args.correlation if correlation is None else correlation,
            randomise_values=args.randomise_values,
            level_dataset=args.levels,
            asynchronous=False,
            seed=seed,
        )
        if args.levels:
            settings["dataset_split"] = args.split
        return MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, run_args, train_state, update = load_train_state(args.checkpoint, env_cfg=env_config(0))
    reader = state_reader_for(policy)
    print(f"checkpoint {args.checkpoint.name}  (update {update})")
    print(
        f"{type(run_args.net).__name__}, {args.size}x{args.size}, rho={args.correlation}, "
        f"{args.train_episodes} train / {args.test_episodes} test episodes\n"
    )

    # An untrained network of the same shape. This, not the observation, is the
    # baseline that decides whether training put the route there: a random
    # convolutional tower already has a receptive field, and that alone beats a
    # pointwise readout of the observation. Built from the checkpoint's own
    # network spec, so it is the same architecture whatever that is.
    untrained = None
    if not args.no_untrained:
        _, _, untrained = run_args.net.init_params(env_config(0).make(), jax.random.PRNGKey(12345))

    test_correlations = args.test_correlations or [args.correlation]

    def rollouts(seed: int, episodes: int, probe_think: int, params, correlation: float | None = None):
        return collect_rollouts(
            env_config(seed, correlation).make(),
            policy,
            train_state.params,
            episodes,
            seed=seed,
            probe_params=params,
            probe_steps_to_think=probe_think,
        )

    print(f"{'think':>6}{'test rho':>9}{'probe':>20}{'AUC':>9}{'95% CI':>16}{'bal.acc':>10}{'episodes':>10}")
    for think in args.think:
        arms: list[tuple[str, ProbeSource, object]] = [
            ("trained", "features", None),
            ("observation", "observation", None),
        ]
        if untrained is not None:
            arms.insert(1, ("untrained", "features", untrained))

        for label, source, params in arms:
            # Fitted once at the training correlation; scored at each test
            # correlation on disjoint seeds, so the probe never sees the levels
            # it is scored on and every scoring arm reads the same probe.
            train = rollouts(0, args.train_episodes, think, params)
            for test_rho in test_correlations:
                test = rollouts(9999, args.test_episodes, think, params, test_rho)

                r = probe(train, test, source=source)
                low, high = auc_interval(train, test, source=source)
                interval = f"[{low:.3f}, {high:.3f}]"
                print(
                    f"{think:>6}{test_rho:>9.2f}{label:>20}{r.auc:>9.3f}{interval:>16}"
                    f"{r.balanced_accuracy:>10.3f}{len(test):>10,}"
                )

                if args.by_distance:
                    bands = probe_by_distance(train, test, source=source)
                    shown = "  ".join(f"{b.step}:{b.auc:.3f}" for b in bands)
                    print(f"        by distance (matched negatives): {shown}")

                if args.per_layer and source == "features":
                    for index, name in enumerate(reader.layer_names):
                        r = probe(only_layer(train, index, reader.n_layers), only_layer(test, index, reader.n_layers))
                        print(
                            f"{'':>6}{test_rho:>9.2f}{name:>20}{r.auc:>9.3f}{'':>16}"
                            f"{r.balanced_accuracy:>10.3f}{'':>10}  (layer only)"
                        )

    print(
        "\nThe trained probe must beat the *untrained* one. Beating only the\n"
        "observation shows the network has a receptive field, which it has\n"
        "before any training at all."
    )


def only_layer(rollouts, index: int, n_layers: int):
    """The same rollouts with one layer's share of the per-cell state.

    Layers are concatenated on the channel axis with equal widths, so a layer
    is a contiguous slice; nothing else about the episode changes.
    """
    out = []
    for r in rollouts:
        width = r.features.shape[-1] // n_layers
        out.append(dataclasses.replace(r, features=r.features[..., index * width : (index + 1) * width]))
    return out


if __name__ == "__main__":
    main()
