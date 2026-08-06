"""Does the network hold a distance field for both objectives, or only its target?

    uv run python experiments/005_outcome_keyed_distance.py CHECKPOINT --levels DIR

The pilot found maze-aware distance in the recurrent state, measured to a fixed
objective. But at test rho=1.0 the agent goes to feature 0 in about three
quarters of episodes, so that measurement was a mixture of "the objective it is
heading to" and "the one it ignored" — and a clean field for the target averaged
with noise for the other looks exactly like a weak field for both.

Keying the same field by outcome separates them, and the answer is mechanistic
either way:

**Both decode** — the network computes a field for each objective and compares
them. That is the substrate the utility comparison would need, and it means the
choice is made *after* the distances exist.

**Only the reached one** — the network commits to a target first and computes
distance second, so the comparison happens somewhere this probe cannot see.

Read the table in this order. The controls come first because they decide
whether anything else is readable: ``oracle`` must pass, ``null`` and
``shuffled`` must fail. Then the reached-minus-unreached difference, which is
computed on a *paired* bootstrap — two overlapping intervals would not settle
it, and the difference interval is far tighter than either alone.
"""

from __future__ import annotations

import argparse
import functools
import operator
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import fields, metrics, targets
from goalmisgen.analysis.activations import Capture, RolloutCache, collect_rollouts, require_one_actor
from goalmisgen.analysis.probes import Feature
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33

N_FEATURES = 2
TAU = 4.0
"""Detour threshold in cells, fixed before any checkpoint was looked at.

At tau=4 roughly 55% of cells on an 11x11 level are hard cells, so the subset is
the majority rather than a thin tail. Sensitivity at 2 and 8 is printed.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--fit-split", type=str, default="valid", help="Probes are fitted here...")
    parser.add_argument("--split", type=str, default="test", help="...and scored here, so no level is shared.")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--test-episodes", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--think", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--randomise-values", action="store_true")
    return parser.parse_args()


def provenance() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}"


def main() -> None:
    args = parse_args()
    print(provenance())

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

    _, _, untrained_params = maze_drc33(min_size=args.size, max_size=args.size).net.init_params(
        env_config(0, args.fit_split).make(), jax.random.PRNGKey(12345)
    )
    registry = {"agent": None, "random": untrained_params}

    def collect(split):
        def run(capture: Capture, seed: int, n_episodes: int):
            return collect_rollouts(
                env_config(seed, split).make(),
                policy,
                train_state.params,
                n_episodes,
                seed=seed,
                probe_params=registry[capture.reader],
                probe_steps_to_think=capture.steps_to_think,
            )

        return RolloutCache(run)

    fit_cache, score_cache = collect(args.fit_split), collect(args.split)

    keyed = {
        "reached": targets.DistanceToObjective(targets.reached, name="d->reached", n_features=N_FEATURES),
        "unreached": targets.DistanceToObjective(
            functools.partial(targets.unreached, n_features=N_FEATURES), name="d->unreached", n_features=N_FEATURES
        ),
        "feature0": targets.DistanceToObjective(targets.fixed(0), name="d->f0", n_features=N_FEATURES),
    }
    activations = Feature("activations", operator.attrgetter("features"))

    header = f"{'capture':>12}{'target':>12}{'arm':>26}{'hard R2':>9}{'95% CI':>18}"
    print(f"\n{header}{'shape':>8}{'partial':>9}{'within':>8}{'MAE':>7}{'dim':>5}{'drop':>6}{'n_hard':>8}")

    for think in args.think:
        captures = [Capture(f"trained@{think}", reader="agent", steps_to_think=think)]
        if think == 0:
            captures.append(Capture("untrained@0", reader="random"))
        require_one_actor(captures)

        for capture in captures:
            fit_rollouts = fit_cache.get(capture, 0, args.train_episodes)
            score_rollouts = score_cache.get(capture, 9999, args.test_episodes)
            paired = {}

            for key, target in keyed.items():
                arms = [activations, *targets.controls(target, score_rollouts)]
                fit_arms = [activations, *targets.controls(target, fit_rollouts)]

                scores = {}
                for arm, fit_arm in zip(arms, fit_arms):
                    train = fields.cell_data(fit_rollouts, fit_arm, target)
                    test = fields.cell_data(score_rollouts, arm, target)
                    result = fields.field_probe(arm.name, train, test, tau=TAU)
                    scores[arm.name.split(":")[0]] = result
                    if arm is activations:
                        _, prediction, _, _ = fields.fit_predict(train, test)
                        paired[key] = (test, prediction)

                    print(
                        f"{capture.name:>12}{key:>12}{result.name:>26}{result.hard_r2:>9.3f}"
                        f"{f'[{result.hard_interval[0]:.3f}, {result.hard_interval[1]:.3f}]':>18}"
                        f"{result.hard_shape_r2:>8.3f}{result.partial_r:>9.3f}"
                        f"{result.partial_r_within_episode:>8.3f}{result.mae:>7.2f}"
                        f"{result.depth:>5}{result.dropped_columns:>6}{result.n_hard:>8,}"
                    )
                print(f"{'':>24}{check_rig(scores)}\n")

            report_difference(paired, capture.name)


def check_rig(scores) -> str:
    """The controls decide whether the row above is readable, so say it aloud.

    Left to a reader to notice, this is the check that gets skipped — the first
    distance-band result was invalidated by a confound raised and waved through.
    """
    missing = {"oracle", "null", "shuffled"} - set(scores)
    if missing:
        raise ValueError(f"a headline was about to be printed without its controls: missing {sorted(missing)}")

    problems = []
    if scores["oracle"].hard_r2 < 0.9:
        problems.append(f"positive control failed ({scores['oracle'].hard_r2:.3f})")
    for name in ("null", "shuffled"):
        if scores[name].hard_r2 > 0.1:
            problems.append(f"{name} control passed ({scores[name].hard_r2:.3f})")

    if problems:
        return "RIG INVALID — " + "; ".join(problems) + ". No number in this block means anything."
    return "controls ok (oracle passes, null and shuffled fail)"


def report_difference(paired, capture: str) -> None:
    """Reached minus unreached, on a common resample of episodes.

    Two overlapping intervals do not settle this. Sharing the resample cancels
    the between-episode variance both statistics carry, and the two targets mask
    different cells, so each statistic looks up its own rows.
    """
    if not {"reached", "unreached"} <= set(paired):
        return

    def hard_r2_over(data, prediction):
        def compute(chosen):
            rows = metrics.select_rows(data.episode, chosen)
            hard = fields.hard_cells(data, TAU)[rows]
            return metrics.r2(data.y[rows][hard], prediction[rows][hard])

        return compute

    low, high = metrics.bootstrap_paired(
        hard_r2_over(*paired["reached"]),
        hard_r2_over(*paired["unreached"]),
        np.unique(paired["reached"][0].episode),
    )
    verdict = "the field is target-specific" if low > 0 else "no detectable difference"
    print(f"{'':>12}reached − unreached hard R²: [{low:+.3f}, {high:+.3f}]   {verdict}   ({capture})\n")


if __name__ == "__main__":
    main()
