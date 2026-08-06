"""Does the network compute a maze-aware distance field, or just diffuse?

    uv run python experiments/004_distance_pilot.py CHECKPOINT --levels DIR

The plan probe shows the agent's future route is linearly decodable. This asks
what that route is built from: does the recurrent state hold, at every cell, an
estimate of that cell's shortest-path distance to an objective? If so, moving is
trivial — step to whichever neighbour holds the lower number — and a 3x3 kernel
iterated nine times per step is exactly the operator that relaxes such a field.

The probe is a 1x1 convolution, so it reads one cell at a time and cannot
combine cells. Anything it recovers is therefore already present *at that cell*;
the probe never computes a distance, it only reads one.

**What would invalidate this.** Iterating 3x3 convolutions spreads the agent and
objective channels outward whether or not the weights mean anything, and that
spread is distance-shaped. So an untrained network of the same architecture may
well decode distance too — the question this pilot exists to settle is whether
it decodes *detours*, where walls force a long way round and isotropic spreading
gives the wrong answer.

Read the table in this order:

1. ``straight-line`` must FAIL the detour test and ``bfs-only`` must PASS it. If
   either is violated the rig is wrong and no other number here means anything.
2. ``observation`` should be near zero: a 1x1 probe sees one cell's channels and
   cannot know where the agent is, so it cannot represent distance at all.
3. Only then compare ``trained`` against ``untrained``, and only on the detour
   column. Pooled R2 is printed for context and is not the claim: a feature with
   no maze solving in it scores about 0.20 there.
"""

from __future__ import annotations

import argparse
import functools
import operator
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import collect_rollouts, fields
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33

N_FEATURES = 2
TAU = 4.0
"""Detour threshold, in cells, fixed before any checkpoint was looked at.

At tau=4 roughly 55% of cells on an 11x11 level are detour cells, so the subset
is the majority rather than a thin tail. Sensitivity at 2 and 8 is printed so it
is visible that this was not tuned.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--test-episodes", type=int, default=512, help="Only ~40 free cells per level, so be generous.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--feature", type=int, default=0, help="Which objective the field is measured to.")
    parser.add_argument("--think", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--randomise-values",
        action="store_true",
        help="Must match the run being probed: the value scheme is part of a dataset's fingerprint.",
    )
    return parser.parse_args()


def straight_line_grid(rollout, feature_id: int) -> np.ndarray:
    """The pre-registered cheat: free-space geometry, no maze solving."""
    _, manhattan, chebyshev = fields.distance_target(rollout, feature_id, N_FEATURES)
    return np.stack([manhattan, chebyshev], axis=-1)


def truth_grid(rollout, feature_id: int) -> np.ndarray:
    """The field itself, handed over. Proves the rig can find one."""
    labels, _, _ = fields.distance_target(rollout, feature_id, N_FEATURES)
    return np.nan_to_num(labels)[:, :, None]


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
    print(f"{args.size}x{args.size}, field = distance to feature {args.feature}, tau = {TAU:g}")
    print(f"{args.train_episodes} train / {args.test_episodes} test episodes\n")

    _, _, untrained = maze_drc33(min_size=args.size, max_size=args.size).net.init_params(
        env_config(0).make(), jax.random.PRNGKey(12345)
    )

    def rollouts(seed: int, episodes: int, think: int, params):
        return collect_rollouts(
            env_config(seed).make(),
            policy,
            train_state.params,
            episodes,
            seed=seed,
            probe_params=params,
            probe_steps_to_think=think,
        )

    activations = operator.attrgetter("features")
    observation = operator.attrgetter("observation")
    straight = functools.partial(straight_line_grid, feature_id=args.feature)
    truth = functools.partial(truth_grid, feature_id=args.feature)

    header = f"{'think':>6}{'arm':>15}{'pooled':>9}{'detour':>9}{'95% CI':>18}{'partial':>9}"
    print(f"{header}{'MAE':>7}{'dim':>6}{'l2':>8}{'|x|':>8}")

    for think in args.think:
        # Trained and untrained need separate collections; the observation and
        # synthetic arms are read off the trained rollouts, since none of them
        # touch the activations.
        trained_train = rollouts(0, args.train_episodes, think, None)
        trained_test = rollouts(9999, args.test_episodes, think, None)
        untrained_train = rollouts(0, args.train_episodes, think, untrained)
        untrained_test = rollouts(9999, args.test_episodes, think, untrained)

        arms = [
            ("trained", activations, trained_train, trained_test),
            ("untrained", activations, untrained_train, untrained_test),
            ("observation", observation, trained_train, trained_test),
            ("bfs-only", truth, trained_train, trained_test),
            ("straight-line", straight, trained_train, trained_test),
        ]

        results = []
        for name, grid, train_rollouts, test_rollouts in arms:
            train = fields.field_dataset(train_rollouts, grid, args.feature, N_FEATURES)
            test = fields.field_dataset(test_rollouts, grid, args.feature, N_FEATURES)
            result = fields.field_probe(name, train, test, tau=TAU)
            results.append(result)
            interval = f"[{result.detour_interval[0]:.3f}, {result.detour_interval[1]:.3f}]"
            print(
                f"{think:>6}{result.name:>15}{result.pooled_r2:>9.3f}{result.detour_r2:>9.3f}{interval:>18}"
                f"{result.partial_r:>9.3f}{result.mae:>7.2f}{result.depth:>6}{result.l2:>8.2g}"
                f"{result.activation_norm:>8.2f}"
            )

            if name in ("trained", "untrained"):
                shown = "  ".join(f"tau={other:g}: {value:.3f}" for other, value in result.sensitivity)
                print(f"{'':>21}{shown}")

        verdict = check_rig(results)
        print(f"\n{verdict}\n")


def check_rig(results) -> str:
    """The controls decide whether the measurement is readable, so say so aloud.

    Left to a reader to notice, this is exactly the check that gets skipped —
    the first distance-band result was invalidated by a confound that had been
    raised and waved through.
    """
    by_name = {r.name: r for r in results}
    problems = []
    if by_name["bfs-only"].detour_r2 < 0.9:
        problems.append(f"positive control failed: bfs-only detour R2 = {by_name['bfs-only'].detour_r2:.3f}")
    if by_name["straight-line"].detour_r2 > 0.1:
        problems.append(f"negative control passed: straight-line detour R2 = {by_name['straight-line'].detour_r2:.3f}")
    if abs(by_name["observation"].partial_r) > 0.25:
        problems.append(f"a 1x1 probe on the observation decoded distance: partial r = {by_name['observation'].partial_r:.3f}")

    if problems:
        return "RIG INVALID — " + "; ".join(problems) + "\nNo other number in this table means anything."
    return (
        "Rig checks passed. The claim is trained vs untrained on the detour column;\n"
        "pooled R2 is context only — straight-line geometry alone scores about 0.20 there."
    )


if __name__ == "__main__":
    main()
