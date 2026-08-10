"""Is what an objective is worth held in a slot, or compiled into a threshold?

    uv run python experiments/014_value_axis_analysis.py \
        --base /workspace/data/runs/novalue11/local-files/cp_140206080 \
        --arms /workspace/data/valueaxis/runs \
        --levels /workspace/data/valueaxis/levels/v050

``013`` fine-tunes the same agent onto a grid of values for colour 1, producing
one weight change per value. This asks what those changes have in common.

The hypothesis is *modularity*, not location. Where a diff lands says very
little — the gradient writes wherever it is cheapest to write, and a sparse diff
is what a sparsity penalty returns whether or not anything modular is in there.
What a family of diffs can show, and a single diff cannot, is the shape of the
map from value to weights:

``collinear``    diffs at different values point the same way, so there is one
                 direction rather than one threshold rebuilt per arm
``graded``       their size tracks how far the value moved
``predictive``   a direction fitted *without* an arm still predicts that arm,
                 in heading and in distance
``writable``     pushing the base weights along it sets values never trained on,
                 and the agent's exchange rate lands where the task's own
                 arithmetic says it should

The last one is the real test, and the one a compiled threshold cannot pass.
Fitting a direction to six points is easy; having it extrapolate beyond them, in
calibrated units of steps, is not.

In-sample fit is reported but should not be believed on its own. Least squares
through the origin is dominated by the arm with the largest offset, so that arm
partly fits itself and scores well even on unrelated diffs — there is a test for
exactly this in ``tests/test_weights.py``. The leave-one-out columns are the
ones that carry the claim.

Everything is scored on one held-out set of levels at the *base* values, so the
readout is identical across arms and no arm is measured on levels it trained on.
The grid shares its mazes — only what the objectives pay differs — so the test
split is common to all of them.

Two controls run alongside: the ``v=0.5`` arm, fine-tuned on the value it
already had, which measures drift rather than value; and norm-matched random
directions, which measure what a perturbation of that size does by itself.
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

from goalmisgen.analysis import collect_episode_outcomes, metrics, summarise
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.weights import cosine, explained, fit_axis, projected_offset
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
        help="Values to write with the fitted axis that no arm was trained on. The defaults "
        "sit outside the grid in both directions, where a threshold compiled from the "
        "training values has no reason to keep working.",
    )
    parser.add_argument("--skip-behaviour", action="store_true", help="Weight-space sections only.")
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
    """Exchange rate in extra steps, its interval, and whether the agent still finishes."""
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
    reached = summarise(outcomes).reached
    print(f"  {label:>30}{point:>8.1f}  [{low:5.1f}, {high:5.1f}]{reached:>11.1%}")
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
    diffs: dict[float, np.ndarray] = {}
    for value, checkpoint in sorted(arm_checkpoints(args.arms).items()):
        _, _, _, state, arm_update = load_train_state(checkpoint, env_cfg=config)
        flat, _ = ravel_pytree(state.params)
        diffs[value] = np.asarray(flat - base_flat, dtype=np.float64)
        print(f"  v={value:.2f}  {checkpoint.name}  (update {arm_update})  |delta| {np.linalg.norm(diffs[value]):.4g}")

    # The null arm was fine-tuned onto the value it already had, so whatever it
    # moved is drift: the same updates on the same task. Held out of the fit and
    # reported against it.
    fitted = {v: d for v, d in diffs.items() if abs(v - BASE_VALUE) > 1e-9}
    null = diffs.get(BASE_VALUE)
    if len(fitted) < 3:
        sys.exit("\nNeed at least three arms away from the base value before any of this means anything.")

    values = sorted(fitted)
    offsets = np.array([v - BASE_VALUE for v in values])
    stacked = np.stack([fitted[v] for v in values])
    axis = fit_axis(offsets, stacked)

    print("\n\n=== collinear? cosine between arms ===\n")
    print("        " + "".join(f"{v:>8.2f}" for v in values))
    for a in values:
        print(f"  {a:.2f}  " + "".join(f"{cosine(fitted[a], fitted[b]):>8.3f}" for b in values))
    print(
        "\nA threshold rebuilt per arm puts these near zero. Arms on the same side of the\n"
        "base should agree, and arms on opposite sides should oppose: a direction that\n"
        "merely means 'was fine-tuned' would make every entry positive instead."
    )

    print("\n=== graded and predictive? ===\n")
    print(f"  {'value':>7}{'offset':>8}{'|delta|':>10}{'in-sample':>11}{'held-out R^2':>14}{'held-out cos':>14}{'implied offset':>16}")
    for value in values:
        others = [v for v in values if v != value]
        held = fit_axis(np.array(others) - BASE_VALUE, np.stack([fitted[v] for v in others]))
        delta, offset = fitted[value], value - BASE_VALUE
        print(
            f"  {value:>7.2f}{offset:>8.2f}{np.linalg.norm(delta):>10.4g}"
            f"{explained(delta, offset, axis):>11.3f}{explained(delta, offset, held):>14.3f}"
            f"{cosine(delta, held):>14.3f}{projected_offset(delta, held):>+16.2f}"
        )
    if null is not None:
        print(f"  {'null':>7}{0.0:>8.2f}{np.linalg.norm(null):>10.4g}{'—':>11}{'—':>14}{cosine(null, axis):>14.3f}{projected_offset(null, axis):>+16.2f}")

    print(
        "\nIn-sample flatters the widest arm, which partly fits itself; the held-out\n"
        "columns are the claim. The implied offset is the axis read backwards — how far\n"
        "this arm moved along a direction it did not help build — and should match the\n"
        "offset it was trained at, which is the axis carrying a scale and not just a\n"
        "heading. The null arm's size is the drift floor: an arm is only carrying value\n"
        "if it moved further than that."
    )

    if args.skip_behaviour:
        return

    print("\n\n=== writable? pushing the base weights along the axis ===\n")
    print(f"  {'':>30}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
    envs = config.make()
    measure(base_state.params, policy, envs, args, "base, untouched")

    rows: list[tuple[float, float]] = []
    for value in sorted(set(values) | set(args.extrapolate)):
        seen = "grid" if value in values else "unseen"
        point, _, _, _ = measure(unravel(base_flat + (value - BASE_VALUE) * axis), policy, envs, args, f"v={value:.2f} written ({seen})")
        rows.append((value, point))

    rng = np.random.default_rng(args.seed)
    for magnitude in (0.3, 0.5):
        direction = rng.normal(size=axis.size)
        direction *= np.linalg.norm(magnitude * axis) / np.linalg.norm(direction)
        measure(unravel(base_flat + direction), policy, envs, args, f"random, matched to {magnitude:.1f}")

    print("\n\n=== calibrated? written value against the task's own exchange rate ===\n")
    print(f"  {'written value':>15}{'measured steps':>17}{'task says':>12}{'error':>9}")
    for value, point in rows:
        expected = (1.0 - value) / args.step_penalty
        print(f"  {value:>15.2f}{point:>17.1f}{expected:>12.1f}{point - expected:>+9.1f}")
    print(
        "\nThe task charges 0.05 a step, so an objective worth v less is worth walking\n"
        "(1 - v) / 0.05 further for. The agent was never told this and the base run\n"
        "already misses it by about two steps, so the axis is not expected to hit it\n"
        "exactly. What matters is the slope — one unit of written value buying the\n"
        "number of steps the arithmetic says it should — and that it holds outside the\n"
        "grid the axis was fitted on."
    )


if __name__ == "__main__":
    main()
