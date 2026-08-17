"""Is the knob a value, or the gap between two values?

    uv run python experiments/015_value_or_gap.py \
        --base /workspace/data/runs/novalue11.s1234/local-files/cp_140206080 \
        --arms /workspace/data/runs/novalue11.s1234/arms \
        --levels /workspace/data/levels/values/1.00-0.50@500k

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

Two traps, one of which this script fell into.

**Attenuation.** Both axes are fitted from diffs that are mostly movement with no
behavioural effect, so each is a noisy estimate of its own direction and the
cosine between them is pulled toward zero whichever hypothesis holds. Dividing by
split-half reliability corrects for that in principle; in practice it multiplied
by nearly seven on this grid, and in ``results/three-objective.txt`` it returned
cosines outside the range a cosine can take, which is a correction announcing
that it has broken down.

So the headline test is a **permutation null** instead. Shuffling which offset
belongs to which diff destroys the association between value and direction while
leaving untouched the large common component every arm carries — the cost of
running the updates, which the null arm measures directly. What comes back is the
distribution of cosines these diffs can produce with no value signal in them,
which is what the observed cosine has to beat. It assumes nothing about how the
noise scales. The reliability correction is still printed, as a secondary
reading.

**``--cross`` does not discriminate.** It was added to settle the question
causally and cannot. Raising colour 0 by ``d`` and lowering colour 1 by ``d``
leave the same gap, so under *both* hypotheses they produce the same policy. No
behavioural test on a two-objective task can separate a value from the gap,
which is what the top of this docstring already said. Separating them needs a
task where the choice does not reduce to one difference — three objectives, where
a single scalar cannot express the problem.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state
from jax.flatten_util import ravel_pytree

from goalmisgen import provenance
from goalmisgen.analysis import collect_episode_outcomes, metrics, summarise
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.weights import cosine, fit_axis_and_drift, permutation_cosines, permutation_p_value
from goalmisgen.configs.env import MazeConfig
from goalmisgen.volume import discover_arms, sweep_index

COLOUR_ZERO_BASE = 1.0
COLOUR_ONE_BASE = 0.5
"""What each objective was worth to the agent before any fine-tuning."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True, help="An agent's arms/ directory, holding both objectives' sweeps.")
    parser.add_argument("--levels", type=str, required=True, help="Levels at the base values; the test split is used.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument(
        "--resamples",
        type=int,
        default=2000,
        help="Shuffles of the offsets against the diffs, building the null the observed "
        "cosine is read against. The p-value cannot go below 1/(resamples+1).",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--at", type=int, default=-1, help="Which checkpoint of each arm, in step order.")
    parser.add_argument(
        "--arm-steps",
        type=int,
        default=None,
        help="Which sweep to read when the agent has been swept at more than one arm "
        "length, e.g. 750000. Arms of different lengths are not comparable.",
    )
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument(
        "--cross",
        action="store_true",
        help="Write colour 0's arms using colour 1's axis with the sign flipped. This does "
        "NOT separate the two hypotheses -- see the note in the module docstring -- and is "
        "kept only as a check that the two axes are interchangeable edits of the gap.",
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


def arm_checkpoints(root: Path, prefix: str, at: int, base_value: float, steps: int | None) -> dict[float, Path]:
    """The chosen checkpoint of every arm in one sweep, keyed by the value it trained at.

    ``steps`` picks which sweep is meant when an agent has been swept more than
    once at different arm lengths, which are not comparable and which
    ``discover_arms`` refuses to mix silently.
    """
    return discover_arms(root, sweep_index(prefix), base_value, steps=steps, at=at)


def load_sweep(
    root: Path, prefix: str, at: int, base_value: float, base_flat, config, steps: int | None = None
) -> dict[float, np.ndarray]:
    """Weight diffs for one sweep, keyed by the value the arm was trained at."""
    diffs: dict[float, np.ndarray] = {}
    for value, checkpoint in sorted(arm_checkpoints(root, prefix, at, base_value, steps).items()):
        _, _, _, state, _ = load_train_state(checkpoint, env_cfg=config)
        flat, _ = ravel_pytree(state.params)
        diffs[value] = np.asarray(flat - base_flat, dtype=np.float64)
        print(
            f"  o{sweep_index(prefix)}{value - base_value:+.2f}  value {value:.2f}  offset {value - base_value:+.2f}  |delta| {np.linalg.norm(diffs[value]):.4g}"
        )
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
    axis, drift = fit_axis_and_drift(np.array(values) - base_value, np.stack([diffs[v] for v in values]))
    return axis, drift, values


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")

    config = eval_config(args)
    policy, _, _, base_state, _ = load_train_state(args.base, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    print(f"base {args.base.name}  ({base_flat.size:,} parameters)\n")

    print("colour 1 swept, colour 0 held at 1.0")
    one = load_sweep(args.arms, "v", args.at, COLOUR_ONE_BASE, base_flat, config, args.arm_steps)
    print("\ncolour 0 swept, colour 1 held at 0.5")
    zero = load_sweep(args.arms, "c", args.at, COLOUR_ZERO_BASE, base_flat, config, args.arm_steps)

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
            f"  {gap:>6.1f}{v_one:>13.2f}{v_zero:>13.2f}" f"{cosine(one[v_one] - drift_one, zero[v_zero] - drift_zero):>14.3f}"
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
        fits = [fit_axis_and_drift(np.array(half) - base_value, np.stack([diffs[v] for v in half]))[0] for half in halves]
        return cosine(*fits)

    reliability_one = split_half(one, values_one, COLOUR_ONE_BASE)
    reliability_zero = split_half(zero, values_zero, COLOUR_ZERO_BASE)
    observed = cosine(axis_zero, axis_one)
    print(f"  split-half reliability of axis_1  {reliability_one:+.3f}")
    print(f"  split-half reliability of axis_0  {reliability_zero:+.3f}")

    # The headline test. Shuffling which offset belongs to which diff destroys
    # the association between value and direction while leaving the drift every
    # arm shares exactly where it is, so the null says what cosine these diffs
    # can produce with no value signal in them. Assuming a null of zero instead
    # would read that shared drift as evidence.
    print("\n=== is the cosine more than these diffs would give anyway? ===\n")
    offsets_one = np.array(values_one) - COLOUR_ONE_BASE
    null = permutation_cosines(
        offsets_one,
        np.stack([one[v] for v in values_one]),
        axis_zero,
        resamples=args.resamples,
        seed=args.seed,
    )
    p_value = permutation_p_value(observed, null, alternative="less")
    print(f"  observed cos(axis_0, axis_1)     {observed:+.3f}")
    print(f"  null over {args.resamples} shuffles         mean {null.mean():+.3f}, sd {null.std():.3f}")
    print(f"  null 5th percentile              {np.percentile(null, 5):+.3f}")
    print(f"  p(null at least this negative)   {p_value:.4f}")
    print(
        "\n  One knob predicts a cosine at -1 and this p-value small. Two value\n"
        "  registers predict an observation sitting inside the null. Nothing here\n"
        "  assumes how the noise scales."
    )

    # Kept as a secondary reading rather than the argument. On the first grid the
    # correction reached x7, and in results/three-objective.txt it returned
    # cosines outside the range a cosine can take.
    if reliability_one > 0 and reliability_zero > 0:
        corrected = observed / np.sqrt(reliability_one * reliability_zero)
        print(f"\n  secondary: cos corrected for attenuation  {corrected:+.3f}")
        if abs(corrected) > 1:
            print("  ...which is outside the range a cosine can take, so the correction has broken down.")

    if args.skip_behaviour:
        return

    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    if args.cross:
        # This was built to separate the hypotheses causally and does not. Under
        # *either* one, lowering colour 1 by d and raising colour 0 by d leave the
        # same gap, and the choice depends on nothing else, so both predict these
        # columns agree. Kept because it does establish something narrower: the
        # two axes are interchangeable as edits of the gap, to within 0.1 steps.
        print("\n\n=== can colour 1's axis write colour 0's arms? ===\n")
        print(f"  {'':>32}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
        measure(base_state.params, policy, get_action, envs, args, "base, untouched")
        checkpoints = arm_checkpoints(args.arms, "c", args.at, COLOUR_ZERO_BASE, args.arm_steps)
        crossed: list[tuple[float, float, float, float]] = []
        for value in values_zero:
            offset = value - COLOUR_ZERO_BASE
            own = measure(
                unravel(base_flat + offset * axis_zero),
                policy,
                get_action,
                envs,
                args,
                f"colour 0 = {value:.2f} via axis_0",
            )
            other = measure(
                unravel(base_flat - offset * axis_one),
                policy,
                get_action,
                envs,
                args,
                f"colour 0 = {value:.2f} via -axis_1",
            )
            _, _, _, arm_state, _ = load_train_state(checkpoints[value], env_cfg=config)
            trained = measure(arm_state.params, policy, get_action, envs, args, f"colour 0 = {value:.2f} fine-tuned")
            crossed.append((value, trained, own, other))

        print(f"\n  {'colour 0':>10}{'fine-tuned':>13}{'via axis_0':>13}{'via -axis_1':>14}")
        for value, trained, own, other in crossed:
            print(f"  {value:>10.2f}{trained:>13.1f}{own:>13.1f}{other:>14.1f}")
        print(
            "\n  These agree under both hypotheses, so this settles nothing about which is\n"
            "  true: the choice depends only on the gap, and both edits move the gap by the\n"
            "  same amount. It does show the two axes are interchangeable edits of it."
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
            unravel(base_flat + (value - COLOUR_ZERO_BASE) * held),
            policy,
            get_action,
            envs,
            args,
            f"colour 0 = {value:.2f} written, held out",
        )
        _, _, _, arm_state, _ = load_train_state(
            arm_checkpoints(args.arms, "c", args.at, COLOUR_ZERO_BASE, args.arm_steps)[value], env_cfg=config
        )
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
