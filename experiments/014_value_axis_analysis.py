"""Is what an objective is worth held in a slot, or compiled into a threshold?

    uv run python experiments/014_value_axis_analysis.py \
        --base /workspace/data/runs/novalue11/local-files/cp_140206080 \
        --arms /workspace/data/valueaxis/runs \
        --levels /workspace/data/valueaxis/levels/v050

``013`` fine-tunes the same agent onto a grid of values for colour 1, producing
one weight change per value. This asks what those changes have in common.

The hypothesis is *modularity*, not location. Where a diff lands says very little
— the gradient writes wherever it is cheapest to write, and a sparse diff is
what an L1 penalty returns whether or not anything modular is in there. The
claim that can actually be tested is about the shape of the map from value to
weights:

``collinear``    diffs at different values point the same way, so there is one
                 direction rather than one threshold rebuilt per arm
``graded``       their size tracks how far the value moved
``predictive``   a direction fitted without an arm still predicts that arm
``writable``     pushing the base weights along it sets values never trained on,
                 and the agent's exchange rate lands where the task's own
                 arithmetic says it should

The last one is the real test, and it is the one a compiled threshold cannot
pass. Fitting a direction to seven points is easy; having it extrapolate to
values outside the grid, in calibrated units of steps, is not.

Everything is measured on one held-out set of levels at the *base* values, so
the readout is identical across arms and no arm is scored on levels it saw. The
levels of the grid are the same mazes throughout — only what the objectives pay
differs — so the test split is shared as well.

Two controls run alongside: the ``v=0.5`` arm, which was fine-tuned on the value
it already had and so measures drift rather than value, and norm-matched random
directions, which measure what any perturbation of that size does.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from functools import partial
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis import collect_episode_outcomes, metrics
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.configs.env import MazeConfig

BASE_VALUE = 0.5
"""What colour 1 was worth to the agent before any fine-tuning."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="The checkpoint every arm was fine-tuned from.")
    parser.add_argument("--arms", type=Path, required=True, help="Directory of 013 run directories, named vXXX.")
    parser.add_argument("--levels", type=str, required=True, help="Levels at the base values; the test split is used.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extrapolate",
        type=float,
        nargs="+",
        default=[1.1, 0.2],
        help="Values to write with the fitted axis that no arm was trained on. "
        "Defaults sit outside the grid in both directions, where a compiled threshold "
        "has no reason to keep working.",
    )
    return parser.parse_args()


def eval_config(args: argparse.Namespace) -> MazeConfig:
    """The base values, the held-out split, and the agent's own observation shape."""
    return MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        feature_value_correlation=1.0,
        objective_values=(1.0, BASE_VALUE),
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )


def arm_checkpoints(root: Path) -> dict[float, Path]:
    """The last checkpoint of every ``vXXX`` run directory under ``root``."""
    found: dict[float, Path] = {}
    for run in sorted(root.iterdir()):
        match = re.fullmatch(r"v(\d{3})", run.name)
        if not match or not (run / "local-files").is_dir():
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            print(f"  {run.name}: no checkpoint saved, skipping")
            continue
        found[int(match.group(1)) / 100] = checkpoints[-1]
    return found


def measure(params, policy, envs, args, label: str) -> tuple[float, float, float, float]:
    """Exchange rate in extra steps, its interval, and how often the agent still finishes."""
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    carry = policy.apply(params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
    state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

    def act(observations, starts):
        state["carry"], action, _, state["key"] = get_action(
            params, state["carry"], observations, starts, state["key"], temperature=0.0
        )
        return np.asarray(action)

    outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
    gaps, took_richer, _ = value_distance_decisions(outcomes)
    point = indifference_point(gaps, took_richer)
    low, high = metrics.bootstrap_episodes(
        lambda rows: indifference_point(gaps[rows], took_richer[rows]),
        np.arange(len(gaps)),
        resamples=200,
        seed=args.seed,
    )
    reached = float(np.mean([o.get("reached", o.get("solved", False)) for o in outcomes]))
    print(f"  {label:>28}  {point:6.1f}  [{low:5.1f}, {high:5.1f}]   reached {reached:6.1%}")
    return point, low, high, reached


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}\n")

    config = eval_config(args)
    policy, _, _, base_state, update = load_train_state(args.base, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    print(f"base {args.base.name}  (update {update}, {base_flat.size:,} parameters)\n")

    print("arms")
    arms = arm_checkpoints(args.arms)
    diffs: dict[float, np.ndarray] = {}
    for value, checkpoint in sorted(arms.items()):
        _, _, _, state, arm_update = load_train_state(checkpoint, env_cfg=config)
        flat, _ = ravel_pytree(state.params)
        diffs[value] = np.asarray(flat - base_flat, dtype=np.float64)
        norm = np.linalg.norm(diffs[value])
        print(f"  v={value:.2f}  {checkpoint.name}  (update {arm_update})  |delta| {norm:.4g}")

    if len(diffs) < 3:
        sys.exit("\nNeed at least three arms before any of this means anything.")

    # The null arm was fine-tuned onto the value it already had. Whatever it
    # moved is drift: the same number of updates on the same task. It is held
    # out of the fit and reported against it.
    fitted = {v: d for v, d in diffs.items() if abs(v - BASE_VALUE) > 1e-9}
    null = diffs.get(BASE_VALUE)

    print("\n\n=== collinear? cosine between arms ===\n")
    values = sorted(fitted)
    print("        " + "".join(f"{v:>8.2f}" for v in values))
    for a in values:
        row = "".join(f"{float(np.dot(fitted[a], fitted[b]) / (np.linalg.norm(fitted[a]) * np.linalg.norm(fitted[b]))):>8.3f}" for b in values)
        print(f"  {a:.2f}  {row}")
    print(
        "\nOne direction rebuilt per arm would put these near zero. Values on the same\n"
        "side of the base share a sign; opposite sides should be anti-correlated if the\n"
        "axis is signed rather than a generic 'was fine-tuned' direction."
    )

    # A slot would make the diff proportional to how far the value moved, so fit
    # the axis by regressing the diffs on that offset through the origin. Scaled
    # to one unit of value, u is then "what changing the value by 1.0 does to
    # the weights", and alpha below is just the value offset itself.
    offsets = np.array([v - BASE_VALUE for v in values])
    stacked = np.stack([fitted[v] for v in values])
    axis = (offsets @ stacked) / float(offsets @ offsets)

    print("\n=== graded? how much of each arm the axis explains ===\n")
    print(f"  {'value':>8}{'offset':>9}{'|delta|':>11}{'R^2 on axis':>14}{'cos to axis':>14}")
    for value in values:
        delta = fitted[value]
        predicted = (value - BASE_VALUE) * axis
        residual = 1 - float(np.sum((delta - predicted) ** 2) / np.sum(delta**2))
        cosine = float(np.dot(delta, axis) / (np.linalg.norm(delta) * np.linalg.norm(axis)))
        print(f"  {value:>8.2f}{value - BASE_VALUE:>9.2f}{np.linalg.norm(delta):>11.4g}{residual:>14.3f}{cosine:>14.3f}")
    if null is not None:
        cosine = float(np.dot(null, axis) / (np.linalg.norm(null) * np.linalg.norm(axis)))
        print(f"  {'null':>8}{0.0:>9.2f}{np.linalg.norm(null):>11.4g}{'—':>14}{cosine:>14.3f}")
        print(
            "\nThe null arm is the same fine-tune with nothing to learn. Its size is the\n"
            "drift floor: an arm is only carrying value if it moved further than this."
        )

    print("\n=== predictive? each arm against an axis fitted without it ===\n")
    for value in values:
        others = [v for v in values if v != value]
        held_offsets = np.array([v - BASE_VALUE for v in others])
        held_axis = (held_offsets @ np.stack([fitted[v] for v in others])) / float(held_offsets @ held_offsets)
        delta = fitted[value]
        cosine = float(np.dot(delta, held_axis) / (np.linalg.norm(delta) * np.linalg.norm(held_axis)))
        scale = float(np.dot(delta, held_axis) / np.dot(held_axis, held_axis))
        print(f"  v={value:.2f}  cos {cosine:>6.3f}   implied offset {scale:>+6.2f}  against the true {value - BASE_VALUE:+.2f}")
    print(
        "\nThe implied offset is the fitted axis read backwards: how far this arm moved\n"
        "along a direction it did not help build. Matching the true offset is the axis\n"
        "carrying a magnitude, not just a heading."
    )

    print("\n\n=== writable? pushing the base weights along the axis ===\n")
    print(f"  {'':>28}  {'steps':>6}  {'95% interval':>14}   {'reached':>14}")
    envs = config.make()
    measure(base_state.params, policy, envs, args, "base, untouched")

    rng = np.random.default_rng(args.seed)
    targets = sorted(set(values) | set(args.extrapolate))
    rows: list[tuple[float, float, float, float]] = []
    for value in targets:
        alpha = value - BASE_VALUE
        params = unravel(base_flat + alpha * axis)
        trained = "trained" if value in values else "unseen "
        point, low, high, _ = measure(params, policy, envs, args, f"v={value:.2f} written  ({trained})")
        rows.append((value, point, low, high))

    direction = rng.normal(size=axis.size)
    direction *= np.linalg.norm(axis) / np.linalg.norm(direction)
    for alpha in (-0.3, 0.3):
        measure(unravel(base_flat + alpha * direction), policy, envs, args, f"random, |alpha|={abs(alpha):.1f}")

    print("\n\n=== calibrated? written value against the task's own exchange rate ===\n")
    print(f"  {'written value':>15}{'measured steps':>17}{'task says':>12}{'error':>9}")
    for value, point, _, _ in rows:
        expected = (1.0 - value) / args.step_penalty
        error = point - expected if np.isfinite(point) else float("nan")
        print(f"  {value:>15.2f}{point:>17.1f}{expected:>12.1f}{error:>+9.1f}")
    print(
        "\nThe task charges 0.05 per step, so an objective worth v less than the other is\n"
        "worth walking (1 - v) / 0.05 further for. The agent was never told this and the\n"
        "base run misses it by about two steps, so the axis is not expected to hit it\n"
        "exactly — what matters is the slope: one unit of written value buying the number\n"
        "of steps the arithmetic says it should, including outside the grid it was fit on."
    )


if __name__ == "__main__":
    main()
