"""Is the knob a value, or the gap between two values?

    uv run python experiments/015_value_or_gap.py \
        --base /workspace/data/runs/novalue11/local-files/cp_140206080 \
        --arms /workspace/data/valueaxis/runs \
        --levels /workspace/data/valueaxis/levels/v050

``014`` found a direction that sets the agent's exchange rate, and showed it
writes values it was never fitted on. That is weaker than it sounds, because on
this task nothing behavioural can tell two hypotheses apart:

``value``      the agent holds what each objective is worth, and the direction
               writes colour 1's
``gap``        the agent holds no values at all, only a threshold on the
               difference in distances, and the direction moves the threshold

Both are a single scalar. Choosing between them behaviourally is impossible
here, because the choice depends only on the difference of the two values, so
any experiment that moves one value can be reproduced by moving the threshold.

The weights can tell them apart. Sweeping *colour 0's* value covers the same
gaps as sweeping colour 1's — colour 0 at 0.6 against colour 1 at 0.5 is the
same 0.1 gap as colour 1 at 0.9 against colour 0 at 1.0 — so under the gap
hypothesis the two sweeps must move the weights along the same direction, and
in opposite senses, since raising colour 0 widens the gap and raising colour 1
narrows it. Under the value hypothesis there are two directions, one per
objective, and they need not be anti-parallel at all.

So the headline number is ``cos(axis_0, axis_1)``:

``about -1``   one knob. What ``014`` found is the threshold, and there is no
               value representation to speak of
``about  0``   two knobs, one per objective, which is what a value slot means

The order matters: a null cosine is only meaningful once ``axis_0`` is shown to
work. A direction fitted from noise would also be uncorrelated with anything, so
this checks that writing along ``axis_0`` reproduces its own arms before any
weight is put on the comparison.
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
from goalmisgen.analysis.weights import cosine, fit_axis_and_drift, projected_offset
from goalmisgen.configs.env import MazeConfig

COLOUR_ZERO_BASE = 1.0
COLOUR_ONE_BASE = 0.5
"""What each objective was worth to the agent before any fine-tuning."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True, help="Holds both the vXXX and cXXX run directories.")
    parser.add_argument("--levels", type=str, required=True, help="Levels at the base values; the test split is used.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--at", type=int, default=-1, help="Which checkpoint of each arm, in step order.")
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument(
        "--cross",
        action="store_true",
        help="Test the one-knob hypothesis causally: write colour 0's arms using colour 1's "
        "axis with the sign flipped. If the two sweeps move one shared knob, that is the "
        "same edit and must reproduce them; a correlation between noisy axes cannot settle "
        "this, since noise attenuates it toward zero whichever hypothesis is true.",
    )
    return parser.parse_args()


def eval_config(args: argparse.Namespace) -> MazeConfig:
    return MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        feature_value_correlation=1.0,
        objective_values=(COLOUR_ZERO_BASE, COLOUR_ONE_BASE),
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )


def arm_checkpoints(root: Path, prefix: str, at: int) -> dict[float, Path]:
    """The chosen checkpoint of every ``<prefix>XXX`` run directory."""
    found: dict[float, Path] = {}
    for run in sorted(root.iterdir()):
        match = re.fullmatch(rf"{prefix}(\d{{3}})", run.name)
        if not match or not (run / "local-files").is_dir():
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if checkpoints:
            try:
                found[int(match.group(1)) / 100] = checkpoints[at]
            except IndexError:
                print(f"  {run.name}: no checkpoint at index {at}, skipping")
    return found


def load_sweep(root: Path, prefix: str, at: int, base_value: float, base_flat, config) -> dict[float, np.ndarray]:
    """Weight diffs for one sweep, keyed by the value the arm was trained at."""
    diffs: dict[float, np.ndarray] = {}
    for value, checkpoint in sorted(arm_checkpoints(root, prefix, at).items()):
        _, _, _, state, _ = load_train_state(checkpoint, env_cfg=config)
        flat, _ = ravel_pytree(state.params)
        diffs[value] = np.asarray(flat - base_flat, dtype=np.float64)
        print(f"  {prefix}{int(round(value * 100)):03d}  value {value:.2f}  offset {value - base_value:+.2f}  |delta| {np.linalg.norm(diffs[value]):.4g}")
    return diffs


def measure(params, policy, get_action, envs, args, label: str) -> float:
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
        lambda rows: indifference_point(gaps[rows], took_richer[rows]), np.arange(len(gaps)), resamples=200, seed=args.seed
    )
    print(f"  {label:>32}{point:>8.1f}  [{low:5.1f}, {high:5.1f}]{summarise(outcomes).reached_objective:>11.1%}")
    return point


def fit(diffs: dict[float, np.ndarray], base_value: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    values = sorted(v for v in diffs if abs(v - base_value) > 1e-9)
    axis, drift = fit_axis_and_drift(
        np.array(values) - base_value, np.stack([diffs[v] for v in values])
    )
    return axis, drift, values


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}\n")

    config = eval_config(args)
    policy, _, _, base_state, _ = load_train_state(args.base, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    print(f"base {args.base.name}  ({base_flat.size:,} parameters)\n")

    print("colour 1 swept, colour 0 held at 1.0")
    one = load_sweep(args.arms, "v", args.at, COLOUR_ONE_BASE, base_flat, config)
    print("\ncolour 0 swept, colour 1 held at 0.5")
    zero = load_sweep(args.arms, "c", args.at, COLOUR_ZERO_BASE, base_flat, config)

    if len(one) < 4 or len(zero) < 4:
        sys.exit("\nBoth sweeps need at least four arms before the comparison means anything.")

    axis_one, drift_one, values_one = fit(one, COLOUR_ONE_BASE)
    axis_zero, drift_zero, values_zero = fit(zero, COLOUR_ZERO_BASE)

    print("\n\n=== the two axes ===\n")
    print(f"  |axis_1|  (per unit of colour 1's value)  {np.linalg.norm(axis_one):.4g}")
    print(f"  |axis_0|  (per unit of colour 0's value)  {np.linalg.norm(axis_zero):.4g}")
    print(f"  |drift_1| {np.linalg.norm(drift_one):.4g}    |drift_0| {np.linalg.norm(drift_zero):.4g}")
    print(f"\n  cos(axis_0, axis_1) = {cosine(axis_zero, axis_one):+.3f}")
    print(
        "\n  about -1  one knob: raising either objective's value moves the same direction\n"
        "            in opposite senses, so what 014 found is the gap, or equivalently a\n"
        "            threshold on the difference in distances, and not a value\n"
        "  about  0  two knobs, one per objective, which is what a value slot means\n"
    )

    print("\n=== the same gap, reached by moving either colour ===\n")
    print(f"  {'gap':>6}{'colour 1 at':>13}{'colour 0 at':>13}{'cos of diffs':>14}")
    for gap in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7):
        v_one, v_zero = round(COLOUR_ZERO_BASE - gap, 2), round(COLOUR_ONE_BASE + gap, 2)
        if v_one not in one or v_zero not in zero:
            continue
        print(
            f"  {gap:>6.1f}{v_one:>13.2f}{v_zero:>13.2f}"
            f"{cosine(one[v_one] - drift_one, zero[v_zero] - drift_zero):>14.3f}"
        )
    print(
        "\n  Under one knob these arms are the same agent reached two ways, so their\n"
        "  diffs should agree once each sweep's common component is removed."
    )

    # A cosine between two noisy estimates is attenuated toward zero whichever
    # hypothesis holds, so it cannot be read without knowing how much of each
    # axis is signal. Splitting a sweep in half and fitting both halves gives
    # that directly: two independent estimates of the *same* axis, whose cosine
    # is its reliability.
    print("\n=== how much of each axis is signal? ===\n")

    def split_half(diffs, values, base_value):
        halves = (values[0::2], values[1::2])
        fits = [
            fit_axis_and_drift(np.array(half) - base_value, np.stack([diffs[v] for v in half]))[0]
            for half in halves
        ]
        return cosine(*fits)

    reliability_one = split_half(one, values_one, COLOUR_ONE_BASE)
    reliability_zero = split_half(zero, values_zero, COLOUR_ZERO_BASE)
    observed = cosine(axis_zero, axis_one)
    print(f"  split-half reliability of axis_1  {reliability_one:+.3f}")
    print(f"  split-half reliability of axis_0  {reliability_zero:+.3f}")
    if reliability_one > 0 and reliability_zero > 0:
        corrected = observed / np.sqrt(reliability_one * reliability_zero)
        print(f"\n  cos corrected for attenuation    {corrected:+.3f}")
        print(
            "\n  Still near -1 after correction means one knob measured through noise.\n"
            "  Near 0 means two. The correction can overshoot past -1 when the halves are\n"
            "  this small, so it bounds rather than settles the question."
        )

    if args.skip_behaviour:
        return

    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    if args.cross:
        # Under one knob the sweeps share a direction: raising colour 0 by d and
        # lowering colour 1 by d are the same edit, so -axis_1 must write colour
        # 0's arms as well as axis_0 does. This is causal, so the noise that
        # attenuates a cosine does not weaken it -- axis_1 has already been shown
        # to work on its own sweep.
        print("\n\n=== can colour 1's axis write colour 0's arms? ===\n")
        print(f"  {'':>32}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
        measure(base_state.params, policy, get_action, envs, args, "base, untouched")
        checkpoints = arm_checkpoints(args.arms, "c", args.at)
        crossed: list[tuple[float, float, float, float]] = []
        for value in values_zero:
            offset = value - COLOUR_ZERO_BASE
            own = measure(
                unravel(base_flat + offset * axis_zero), policy, get_action, envs, args,
                f"colour 0 = {value:.2f} via axis_0",
            )
            other = measure(
                unravel(base_flat - offset * axis_one), policy, get_action, envs, args,
                f"colour 0 = {value:.2f} via -axis_1",
            )
            _, _, _, arm_state, _ = load_train_state(checkpoints[value], env_cfg=config)
            trained = measure(arm_state.params, policy, get_action, envs, args, f"colour 0 = {value:.2f} fine-tuned")
            crossed.append((value, trained, own, other))

        print(f"\n  {'colour 0':>10}{'fine-tuned':>13}{'via axis_0':>13}{'via -axis_1':>14}")
        for value, trained, own, other in crossed:
            print(f"  {value:>10.2f}{trained:>13.1f}{own:>13.1f}{other:>14.1f}")
        print(
            "\n  One knob predicts the last two columns agree, since they would be the same\n"
            "  edit written two ways. Two slots predicts colour 1's axis does not set\n"
            "  colour 0's value, and the last column departs from the other two."
        )
        return

    print("\n\n=== does axis_0 work at all? held-out writes ===\n")
    print(f"  {'':>32}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    measure(base_state.params, policy, get_action, envs, args, "base, untouched")

    rows: list[tuple[float, float, float]] = []
    for value in values_zero:
        others = [v for v in values_zero if v != value]
        held, _ = fit_axis_and_drift(np.array(others) - COLOUR_ZERO_BASE, np.stack([zero[v] for v in others]))
        written = measure(
            unravel(base_flat + (value - COLOUR_ZERO_BASE) * held), policy, get_action, envs, args,
            f"colour 0 = {value:.2f} written, held out",
        )
        _, _, _, arm_state, _ = load_train_state(arm_checkpoints(args.arms, "c", args.at)[value], env_cfg=config)
        trained = measure(arm_state.params, policy, get_action, envs, args, f"colour 0 = {value:.2f} fine-tuned")
        rows.append((value, trained, written))

    print(f"\n  {'colour 0':>10}{'gap':>7}{'fine-tuned':>13}{'written':>10}{'error':>9}")
    for value, trained, written in rows:
        print(f"  {value:>10.2f}{value - COLOUR_ONE_BASE:>7.2f}{trained:>13.1f}{written:>10.1f}{written - trained:>+9.1f}")
    print(
        "\n  This is the precondition for reading the cosine above. An axis fitted from\n"
        "  noise would be uncorrelated with anything, so a null cosine only says\n"
        "  something once axis_0 is shown to reproduce arms it never saw."
    )


if __name__ == "__main__":
    main()
