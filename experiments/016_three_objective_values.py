"""With three objectives, does the agent hold a value per objective?

    uv run python experiments/016_three_objective_values.py \
        --base /workspace/data/threeobj/runs/base/local-files/cp_XXXXXXXX \
        --arms /workspace/data/threeobj/runs \
        --levels /workspace/data/threeobj/levels/base

``015`` could not decide between a value per objective and a single threshold on
the distance gap, because with two objectives the choice turns on one difference
and the two hypotheses predict the same policy. Three objectives break that tie,
but not in the way rank alone would suggest: an agent that solves a three-way
choice at all must depend on two independent differences, so finding two
dimensions is close to forced by the task.

What is not forced is **composition**. If each objective's worth is held
separately, then moving two of them is the sum of moving each, and an arm
trained with both moved is reproduced by adding two axes neither of which was
fitted on it. If instead each configuration of values was solved on its own
terms, the sum predicts nothing. That is the test this script is built around,
and the ``m_`` arms exist only to be held out of the fit.

Three further readings come almost free:

``reliability``   each objective is swept at offsets -0.2, -0.1, +0.1, +0.2, so
                  two disjoint symmetric pairs give two independent estimates of
                  the same axis. Their cosine is how much of an axis is signal,
                  which is what ``015`` lacked and what made its cosine
                  uninterpretable
``sum``           adding a constant to every value changes no choice, so a
                  representation that holds only differences should have
                  ``axis_0 + axis_1 + axis_2`` near zero, while absolute
                  per-objective registers need not
``pairwise``      whether the three axes are distinct directions at all, read
                  against the reliability ceiling rather than against 1

Offsets are symmetric about the base on purpose. In the first grid they ran
-0.2 to +0.4, and that imbalance let the common fine-tuning component leak into
the fitted axis and produce a confident-looking null.
"""

from __future__ import annotations

import argparse
import itertools
import re
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
from goalmisgen.analysis.weights import cosine, fit_axis_and_drift
from goalmisgen.configs.env import MazeConfig

BASE_VALUES = (1.0, 0.65, 0.3)
"""What the base agent was trained at. Overridden by --base-values, since the
grid was rerun on (1.0, 0.55, 0.4) and a hardcoded triple would silently
mislabel every arm's offset."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--levels", type=str, required=True, help="Levels at the base values; the test split is used.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--at", type=int, default=-1)
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument("--base-values", type=float, nargs="+", default=None, help="What the base agent was trained at.")
    args = parser.parse_args()
    if args.base_values:
        global BASE_VALUES
        BASE_VALUES = tuple(args.base_values)
    return args


def eval_config(args: argparse.Namespace) -> MazeConfig:
    return MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        n_objectives=3,
        objective_values=BASE_VALUES,
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )


def arm_values(name: str) -> tuple[float, ...] | None:
    """The value triple an arm was trained at, read from its directory name.

    ``o1_075`` moves objective 1 to 0.75 and leaves the others at base;
    ``m_110_055_030`` gives all three outright. Reading them from the name keeps
    the analysis honest if a grid is edited: a table here could drift out of
    step with what was actually run, a directory name cannot.
    """
    if (single := re.fullmatch(r"o(\d)_(\d{3})", name)) is not None:
        index, value = int(single.group(1)), int(single.group(2)) / 100
        values = list(BASE_VALUES)
        values[index] = value
        return tuple(values)
    if (mixed := re.fullmatch(r"m(?:_(\d{3})){3}", name)) is not None:
        del mixed
        return tuple(int(part) / 100 for part in name.split("_")[1:])
    return None


def load_arms(root: Path, at: int, base_flat, config) -> dict[str, tuple[tuple[float, ...], np.ndarray]]:
    arms: dict[str, tuple[tuple[float, ...], np.ndarray]] = {}
    for run in sorted(root.iterdir()):
        values = arm_values(run.name)
        if values is None or not (run / "local-files").is_dir():
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            print(f"  {run.name}: no checkpoint, skipping")
            continue
        _, _, _, state, _ = load_train_state(checkpoints[at], env_cfg=config)
        flat, _ = ravel_pytree(state.params)
        diff = np.asarray(flat - base_flat, dtype=np.float64)
        arms[run.name] = (values, diff)
        offsets = tuple(round(v - b, 2) for v, b in zip(values, BASE_VALUES))
        print(f"  {run.name:>16}  values {values}  offset {offsets}  |delta| {np.linalg.norm(diff):.4g}")
    return arms


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
    summary = summarise(outcomes)
    print(f"  {label:>34}{point:>8.1f}   optimal {summary.chose_optimal:>6.1%}   reached {summary.reached_objective:>6.1%}")
    return point


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")

    config = eval_config(args)
    policy, _, _, base_state, _ = load_train_state(args.base, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    print(f"base {args.base.name}  ({base_flat.size:,} parameters)  values {BASE_VALUES}\n")

    print("arms")
    arms = load_arms(args.arms, args.at, base_flat, config)
    single = {name: v for name, v in arms.items() if name.startswith("o")}
    mixed = {name: v for name, v in arms.items() if name.startswith("m")}

    axes: dict[int, np.ndarray] = {}
    reliability: dict[int, float] = {}
    for index in range(3):
        rows = [
            (values[index] - BASE_VALUES[index], diff)
            for values, diff in single.values()
            if abs(values[index] - BASE_VALUES[index]) > 1e-9
            and all(abs(v - b) < 1e-9 for j, (v, b) in enumerate(zip(values, BASE_VALUES)) if j != index)
        ]
        if len(rows) < 3:
            print(f"\nobjective {index}: only {len(rows)} arms, cannot fit an axis")
            continue
        offsets = np.array([o for o, _ in rows])
        axes[index], _ = fit_axis_and_drift(offsets, np.stack([d for _, d in rows]))

        # Two disjoint symmetric pairs, each giving an axis by finite difference.
        # Independent estimates of the same thing, so their cosine says how much
        # of the fitted axis is signal rather than fine-tuning noise.
        #
        # The pairs are found rather than assumed. They were once written out as
        # (0.2, -0.2) and (0.1, -0.1), which silently reported nothing at all the
        # moment a grid used different offsets -- and a grid gets widened exactly
        # when reliability is the thing under investigation.
        table = {round(o, 3): d for o, d in rows}
        pairs = [(m, -m) for m in sorted({abs(o) for o in table}, reverse=True) if m in table and -m in table]
        if len(pairs) >= 2:
            estimates = [(table[hi] - table[lo]) / (hi - lo) for hi, lo in pairs[:2]]
            reliability[index] = cosine(*estimates)
        else:
            print(f"  objective {index}: {len(pairs)} symmetric pair(s), need two to estimate reliability")

    if len(axes) < 3:
        sys.exit("\nNeed an axis for every objective before any of this means anything.")

    print("\n\n=== how much of each axis is signal? ===\n")
    for index in range(3):
        print(
            f"  objective {index}: |axis| {np.linalg.norm(axes[index]):>9.4g}   split-half reliability {reliability.get(index, float('nan')):+.3f}"
        )
    print(
        "\n  Two disjoint symmetric pairs of arms give two estimates of the same axis.\n"
        "  Everything below is attenuated by whatever this falls short of 1."
    )

    # Split-half reliability describes a *half-length* estimate, and the axes
    # being correlated are fitted on all the arms. Dividing a full-axis cosine by
    # the half-length figure over-corrects, which is what put every corrected
    # value outside the range a cosine can occupy. Spearman-Brown converts one to
    # the other.
    full = {index: 2 * r / (1 + r) if r > -1 else float("nan") for index, r in reliability.items()}
    print("\n=== are the three axes distinct directions? ===\n")
    print(f"  {'pair':>10}{'cosine':>10}{'corrected':>12}   ceiling from")
    for i, j in itertools.combinations(range(3), 2):
        raw = cosine(axes[i], axes[j])
        floor = full.get(i, float("nan")) * full.get(j, float("nan"))
        corrected = raw / np.sqrt(floor) if floor > 0 else float("nan")
        flag = "  <- too noisy to read" if min(full.get(i, 0), full.get(j, 0)) < 0.1 else ""
        print(
            f"  {f'{i} vs {j}':>10}{raw:>10.3f}{corrected:>12.3f}   {full.get(i, float('nan')):.3f}, {full.get(j, float('nan')):.3f}{flag}"
        )
    print(
        "\n  One shared knob puts every pair at -1; a representation holding only the\n"
        "  differences puts them at -0.5, since three symmetric vectors summing to zero\n"
        "  must; three absolute registers put them near 0. A corrected figure still\n"
        "  outside the range a cosine can take means the ceiling is too small to divide by."
    )

    total = axes[0] + axes[1] + axes[2]
    scale = float(np.mean([np.linalg.norm(axes[i]) for i in range(3)]))
    print(f"\n  |axis_0 + axis_1 + axis_2| / mean|axis| = {np.linalg.norm(total) / scale:.3f}")
    print(
        "  Adding a constant to every value changes no choice, so a representation that\n"
        "  holds only differences should put this near zero, and absolute per-objective\n"
        "  registers need not. Three unrelated directions would give about 1.7."
    )

    if not mixed:
        print("\n\nNo m_ arms present yet, so composition -- the test that discriminates -- has not run.")
        return

    print("\n\n=== composition: two values moved at once ===\n")
    print(f"  {'arm':>16}{'offsets':>22}{'cos(sum, actual)':>19}{'|sum|/|actual|':>16}")
    for name, (values, diff) in sorted(mixed.items()):
        offsets = np.array([v - b for v, b in zip(values, BASE_VALUES)])
        predicted = sum(offsets[i] * axes[i] for i in range(3))
        print(
            f"  {name:>16}{str(tuple(round(o, 2) for o in offsets)):>22}"
            f"{cosine(predicted, diff):>19.3f}{np.linalg.norm(predicted) / np.linalg.norm(diff):>16.3f}"
        )
    print(
        "\n  These arms were held out of every fit. A value held per objective makes the\n"
        "  sum of two single-value axes the right prediction for moving both; a policy\n"
        "  rebuilt per configuration makes it the wrong one. Read the cosine against the\n"
        "  reliability ceiling above, not against 1."
    )

    if args.skip_behaviour:
        return

    print("\n\n=== and does the composed edit behave like the arm? ===\n")
    print(f"  {'':>34}{'steps':>8}   {'optimal':>14}   {'reached':>14}")
    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    measure(base_state.params, policy, get_action, envs, args, "base, untouched")
    for name, (values, _) in sorted(mixed.items()):
        offsets = [v - b for v, b in zip(values, BASE_VALUES)]
        written = base_flat + sum(offsets[i] * axes[i] for i in range(3))
        measure(unravel(written), policy, get_action, envs, args, f"{name} composed")
        _, _, _, arm_state, _ = load_train_state(
            sorted((args.arms / name / "local-files").glob("cp_*"))[args.at], env_cfg=config
        )
        measure(arm_state.params, policy, get_action, envs, args, f"{name} fine-tuned")
    print(
        "\n  The exchange rate is a two-objective summary and three objectives do not\n"
        "  reduce to one, so read `optimal` alongside it: an edit that sets the values\n"
        "  right should pick the best objective as often as the arm trained to."
    )


if __name__ == "__main__":
    main()
