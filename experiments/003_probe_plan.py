"""Is the agent's route decodable from its hidden state before it moves?

    uv run python experiments/003_probe_plan.py CHECKPOINT --levels DIR --size 11

This is the planning-interpretability result reproduced in our environment. A
linear readout, applied identically at every maze cell, tries to predict which
cells the agent will step on — from the DRC's state at t=0, before it has taken
an action.

The comparison that carries the claim is against the **same probe fitted to the
raw observation**. That input contains the walls and both objectives but no
route, and a linear map cannot search a maze. So:

* observation probe near chance, activation probe well above it
  → the route is represented, not computed by the probe
* both high
  → the label is predictable from layout alone and the probe proves nothing
* both near chance
  → no decodable plan, which would be the interesting negative result

``--think`` re-applies the network to the initial observation before probing,
without stepping the environment. If plans are refined iteratively across the
recurrent ticks, accuracy should climb with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import collect_rollouts, probe, probe_by_distance
from goalmisgen.analysis.probes import ProbeSource
from goalmisgen.configs.env import MazeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--correlation", type=float, default=1.0)
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
            level_dataset=args.levels,
            asynchronous=False,
            seed=seed,
        )
        if args.levels:
            settings["dataset_split"] = args.split
        return MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=env_config(0))
    print(f"checkpoint {args.checkpoint.name}  (update {update})")
    print(
        f"{args.size}x{args.size}, rho={args.correlation}, "
        f"{args.train_episodes} train / {args.test_episodes} test episodes\n"
    )

    print(f"{'think':>6}{'probe':>14}{'AUC':>9}{'bal.acc':>10}{'n cells':>10}")
    for think in args.think:
        # Disjoint seeds so the probe is scored on levels it was not fitted on.
        train = collect_rollouts(
            env_config(0).make(), policy, train_state.params, args.train_episodes, seed=0, steps_to_think=think
        )
        test = collect_rollouts(
            env_config(9999).make(), policy, train_state.params, args.test_episodes, seed=9999, steps_to_think=think
        )

        sources: tuple[tuple[ProbeSource, str], ...] = (("features", "hidden state"), ("observation", "observation"))
        for source, label in sources:
            r = probe(train, test, source=source)
            print(f"{think:>6}{label:>14}{r.auc:>9.3f}{r.balanced_accuracy:>10.3f}{r.n_samples:>10,}")

        if args.by_distance:
            print(f"\n  think={think}, by distance along the route (AUC vs never-visited cells):")
            print(f"  {'step':>6}{'hidden':>9}{'obs':>9}{'n cells':>10}")
            hidden = probe_by_distance(train, test, source="features")
            observation = {b.step: b for b in probe_by_distance(train, test, source="observation")}
            for band in hidden:
                other = observation.get(band.step)
                obs_auc = f"{other.auc:.3f}" if other else "-"
                print(f"  {band.step:>6}{band.auc:>9.3f}{obs_auc:>9}{band.n_positive:>10,}")
            print()

    print(
        "\nThe hidden-state probe must beat the observation probe. If it does not,\n"
        "the label is readable from layout alone and nothing about planning follows."
    )


if __name__ == "__main__":
    main()
