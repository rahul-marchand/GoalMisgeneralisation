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
from goalmisgen.analysis.weights import cosine, explained, fit_axis_and_drift, projected_offset
from goalmisgen.configs.env import MazeConfig

BASE_VALUE = 0.5  # default; --base-value overrides for the colour-0 sweep
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
        "--prefix",
        default="v",
        help="Run-directory prefix naming the sweep: 'v' moves colour 1, 'c' moves colour 0. "
        "The two are the same analysis pointed at a different objective.",
    )
    parser.add_argument(
        "--base-value",
        type=float,
        default=None,
        help="What the swept objective is worth in the untouched agent, so offsets are "
        "measured from it. Defaults to 0.5 for the colour-1 sweep, 1.0 for colour 0.",
    )
    parser.add_argument(
        "--extrapolate",
        type=float,
        nargs="+",
        default=[1.1, 0.2],
        help="Values to write with the fitted axis that no arm was trained on. The defaults "
        "sit outside the grid in both directions, where a threshold compiled from the "
        "training values has no reason to keep working.",
    )
    parser.add_argument(
        "--at",
        type=int,
        default=-1,
        help="Which saved checkpoint of each arm to use, as an index into the arm's "
        "checkpoints in step order; -1 is the last. Behaviour converges well before the "
        "end of a fine-tune, so every update after that adds drift to the diff without "
        "adding value. Reading the grid at an earlier, matched budget is how to see "
        "whether the axis is the same direction with less of it.",
    )
    parser.add_argument("--skip-behaviour", action="store_true", help="Weight-space sections only.")
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help="Write each grid value from an axis fitted without that arm, and compare against "
        "what the arm itself learned. Writing a value the axis was fitted on is in-sample and "
        "cannot separate a real axis from one that memorised the arms it was built from.",
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
        objective_values=(1.0, 0.5),
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )


def arm_checkpoints(root: Path, at: int = -1, prefix: str = "v") -> dict[float, Path]:
    """One checkpoint from every ``<prefix>XXX`` run directory under ``root``.

    Arms are only comparable when read at the same number of updates, so the
    caller picks by index and the step each arm resolved to is printed. An arm
    that saved a different number of checkpoints would otherwise be silently
    compared at a different budget from the rest.
    """
    found: dict[float, Path] = {}
    for run in sorted(root.iterdir()):
        match = re.fullmatch(rf"{prefix}(\d{{3}})", run.name)
        if not match or not (run / "local-files").is_dir():
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            print(f"  {run.name}: no checkpoint saved, skipping")
            continue
        try:
            found[int(match.group(1)) / 100] = checkpoints[at]
        except IndexError:
            print(f"  {run.name}: only {len(checkpoints)} checkpoints, no index {at}, skipping")
    return found


def measure(params, policy, get_action, envs, args, label: str) -> tuple[float, float, float, float]:
    """Exchange rate in extra steps, its interval, and whether the agent still finishes.

    ``get_action`` is compiled once by the caller and reused. Params are an
    argument rather than a constant, so every arm and every written value shares
    one compilation; jitting inside here instead would recompile for each of the
    dozen measurements and cost more than the rollouts do.
    """
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
    reached = summarise(outcomes).reached_objective
    print(f"  {label:>30}{point:>8.1f}  [{low:5.1f}, {high:5.1f}]{reached:>11.1%}")
    return point, low, high, reached


def main() -> None:
    global BASE_VALUE
    args = parse_args()
    BASE_VALUE = args.base_value if args.base_value is not None else (1.0 if args.prefix == "c" else 0.5)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}\n")

    config = eval_config(args)
    policy, _, _, base_state, update = load_train_state(args.base, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    print(f"base {args.base.name}  (update {update}, {base_flat.size:,} parameters)\n")

    print("arms")
    selected = arm_checkpoints(args.arms, args.at, args.prefix)
    steps = {int(path.name.removeprefix("cp_")) for path in selected.values()}
    if len(steps) > 1:
        print(f"  WARNING: arms are at different budgets {sorted(steps)}, so their diffs are not comparable")
    diffs: dict[float, np.ndarray] = {}
    for value, checkpoint in sorted(selected.items()):
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
    axis, drift = fit_axis_and_drift(offsets, stacked)

    # Every arm moved by roughly the same amount whatever it was trained on, and
    # the null arm moved by that amount with nothing to learn, so most of a diff
    # is the cost of 585 updates rather than the value. What is left after the
    # common component is removed is the part that could carry a value at all.
    residual = {v: fitted[v] - drift for v in values}
    print(f"\ncommon component |drift| {np.linalg.norm(drift):.4g}   axis per unit value |axis| {np.linalg.norm(axis):.4g}")
    if null is not None:
        print(f"null arm |delta| {np.linalg.norm(null):.4g}, and {cosine(null, drift):.3f} of it is that same common direction")

    print("\n\n=== collinear? cosine between arms, raw ===\n")
    print("        " + "".join(f"{v:>8.2f}" for v in values))
    for a in values:
        print(f"  {a:.2f}  " + "".join(f"{cosine(fitted[a], fitted[b]):>8.3f}" for b in values))

    print("\n=== and with the common component removed ===\n")
    print("        " + "".join(f"{v:>8.2f}" for v in values))
    for a in values:
        print(f"  {a:.2f}  " + "".join(f"{cosine(residual[a], residual[b]):>8.3f}" for b in values))
    print(
        "\nThe raw matrix is positive everywhere, including between arms on opposite\n"
        "sides of the base, which is the signature of a direction that means no more\n"
        "than 'was fine-tuned'. Once that is removed, arms on the same side should\n"
        "agree and arms on opposite sides should oppose. Ordering matters as much as\n"
        "sign: neighbouring values should look more alike than distant ones."
    )

    print("\n=== graded and predictive? ===\n")
    print(f"  {'value':>7}{'offset':>8}{'|delta|':>10}{'|residual|':>12}{'held-out R^2':>14}{'held-out cos':>14}{'implied offset':>16}")
    for value in values:
        others = [v for v in values if v != value]
        held_axis, held_drift = fit_axis_and_drift(
            np.array(others) - BASE_VALUE, np.stack([fitted[v] for v in others])
        )
        offset = value - BASE_VALUE
        left = fitted[value] - held_drift
        print(
            f"  {value:>7.2f}{offset:>8.2f}{np.linalg.norm(fitted[value]):>10.4g}{np.linalg.norm(residual[value]):>12.4g}"
            f"{explained(left, offset, held_axis):>14.3f}{cosine(left, held_axis):>14.3f}"
            f"{projected_offset(left, held_axis):>+16.2f}"
        )
    if null is not None:
        left = null - drift
        print(f"  {'null':>7}{0.0:>8.2f}{np.linalg.norm(null):>10.4g}{np.linalg.norm(left):>12.4g}{'—':>14}{cosine(left, axis):>14.3f}{projected_offset(left, axis):>+16.2f}")
    print(
        "\nThe implied offset is the axis read backwards, fitted without this arm. It\n"
        "should track the offset the arm was trained at, and the null arm's should sit\n"
        "near zero. All of them landing on the same number is the fit reporting the\n"
        "common component rather than the value."
    )

    if args.skip_behaviour:
        return

    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    if args.leave_one_out:
        print("\n\n=== writable out of sample? each value from an axis that never saw it ===\n")
        print(f"  {'':>30}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
        out_of_sample: list[tuple[float, float, float]] = []
        for value in values:
            others = [v for v in values if v != value]
            held_axis, _ = fit_axis_and_drift(np.array(others) - BASE_VALUE, np.stack([fitted[v] for v in others]))
            written_point, _, _, _ = measure(
                unravel(base_flat + (value - BASE_VALUE) * held_axis), policy, get_action, envs, args, f"v={value:.2f} written, held out"
            )
            _, _, _, arm_state, _ = load_train_state(selected[value], env_cfg=config)
            arm_point, _, _, _ = measure(arm_state.params, policy, get_action, envs, args, f"v={value:.2f} fine-tuned")
            out_of_sample.append((value, arm_point, written_point))

        print(f"\n  {'value':>8}{'fine-tuned':>13}{'written':>10}{'error':>9}")
        for value, arm_point, written_point in out_of_sample:
            print(f"  {value:>8.2f}{arm_point:>13.1f}{written_point:>10.1f}{written_point - arm_point:>+9.1f}")
        print(
            "\nEach written value here comes from an axis fitted on the other five arms only,\n"
            "so nothing about this arm went into the direction that reproduces it. This is\n"
            "the comparison that separates an axis from a lookup of the arms it was built\n"
            "from; the in-sample version cannot."
        )
        return

    print("\n\n=== what the arms themselves do ===\n")
    print(f"  {'':>30}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
    measure(base_state.params, policy, get_action, envs, args, "base, untouched")
    arm_points: list[tuple[float, float]] = []
    for value, checkpoint in sorted(selected.items()):
        _, _, _, state, _ = load_train_state(checkpoint, env_cfg=config)
        point, _, _, _ = measure(state.params, policy, get_action, envs, args, f"v={value:.2f} fine-tuned")
        arm_points.append((value, point))

    print("\n\n=== writable? pushing the base weights along the axis ===\n")
    print(f"  {'':>30}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
    written: list[tuple[float, float]] = []
    for value in sorted(set(values) | set(args.extrapolate)):
        seen = "grid" if value in values else "unseen"
        params = unravel(base_flat + (value - BASE_VALUE) * axis)
        point, _, _, _ = measure(params, policy, get_action, envs, args, f"v={value:.2f} written ({seen})")
        written.append((value, point))

    rng = np.random.default_rng(args.seed)
    for magnitude in (0.3, 0.5):
        direction = rng.normal(size=axis.size)
        direction *= np.linalg.norm(magnitude * axis) / np.linalg.norm(direction)
        measure(unravel(base_flat + direction), policy, get_action, envs, args, f"random, matched to {magnitude:.1f}")

    print("\n\n=== calibrated? against the task's own exchange rate ===\n")
    print(f"  {'value':>8}{'task says':>12}{'fine-tuned':>13}{'written':>10}")
    trained = dict(arm_points)
    for value, point in written:
        expected = (1.0 - value) / args.step_penalty
        actual = trained.get(value)
        print(f"  {value:>8.2f}{expected:>12.1f}{(f'{actual:.1f}' if actual is not None else '—'):>13}{point:>10.1f}")
    print(
        "\nThe fine-tuned column is what an arm actually learned; the written column is\n"
        "what the axis reproduces without training. They agree only if the axis is\n"
        "carrying the value, and the slope is what to read rather than the absolute\n"
        "number: the base agent already undervalues distance by about a fifth, and\n"
        "every arm inherits that."
    )


if __name__ == "__main__":
    main()
