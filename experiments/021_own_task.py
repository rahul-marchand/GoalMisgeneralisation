"""Is each arm actually good at the task it was fine-tuned on?

    uv run python experiments/021_own_task.py \
        --arms /workspace/data/threeobj2/runs --levels /workspace/data/threeobj2/levels \
        --base /workspace/data/threeobj/runs/base/local-files/cp_70103040 \
        --base-levels /workspace/data/threeobj/levels/base --base-values 1.0 0.65 0.3

Every three-objective grid so far produced three collinear axes, and the
explanation on offer is that the base task admitted a one-parameter solution and
the agent took it. With values evenly spaced, each pairwise threshold is a rank
gap times one constant, and the rank comes free from the colour channels — so a
single stored number solves the task, and a short fine-tune can only move the
number that exists.

That story makes a behavioural prediction, and this is it. Moving any single
value **breaks** the even spacing: (1.0, 0.65, 0.3) with objective 1 lowered by
0.2 becomes gaps of 0.55 and 0.15, which no single constant can express. So an
arm should be *measurably suboptimal on its own task*, and the shortfall should
grow with how far its values sit from an arithmetic progression.

The point of asking it this way is that it touches no weight-space quantity. No
axis, no cosine, no reliability — none of the estimators that have proved
delicate. It is just: how often does each agent pick the best objective, on the
task it was trained for, on levels it never saw.

Two outcomes, both worth having:

* shortfall grows with asymmetry — the projection account is right, established
  independently of anything measured in weight space
* arms are near-optimal on their own tasks — they have the second degree of
  freedom after all, and the weight-space measurement is what is wrong

Asymmetry is computed on values sorted high to low, because feature ids are
assigned by rank: an arm whose nominal values put objective 1 below objective 2
still has colour 1 marking the middle-valued objective.

**Asymmetry alone is not enough, and the first run showed why.** How often an
agent picks the best objective depends overwhelmingly on how far ahead the best
one is: on the (1.0, 0.65, 0.3) grid, the gap between the top two values
correlates with the score at r = +0.85, while asymmetry on its own reaches only
r = -0.20. Near-tied leaders are simply a harder task, whatever the agent holds.

Controlling for that separates them, since the two regressors are nearly
independent (r = +0.13):

    optimal = 80.7 + 22.7 * top_gap - 9.0 * asymmetry     residual sd 2.6 points

giving a partial correlation of -0.59 for asymmetry. That is the projection
account supported behaviourally. It is also a covariate chosen after seeing the
table, on eighteen arms, so it generates the hypothesis rather than confirming
it.

**Pre-registered prediction, recorded before the (1.0, 0.55, 0.4) grid finishes
training.** Fitting the same two-regressor model there should give a negative
asymmetry coefficient, of the same order as -9 points per unit, with top_gap
again the larger term. A coefficient near zero on that grid refutes the
projection account, and would mean the collinear axes measured on the first two
grids are a fact about the measurement rather than about the agent.
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

from goalmisgen.analysis import collect_episode_outcomes, summarise
from goalmisgen.configs.env import MazeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--levels", type=Path, required=True, help="Directory holding one dataset per arm.")
    parser.add_argument("--base", type=Path, required=True, help="The checkpoint the arms were fine-tuned from.")
    parser.add_argument("--base-levels", type=str, required=True)
    parser.add_argument("--base-values", type=float, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def values_of(tag: str, base: tuple[float, ...]) -> tuple[float, ...] | None:
    """The value triple an arm trained at, from its directory name."""
    if (single := re.fullmatch(r"o(\d)_(\d{3})", tag)) is not None:
        values = list(base)
        values[int(single.group(1))] = int(single.group(2)) / 100
        return tuple(values)
    if re.fullmatch(r"m(?:_\d{3}){%d}" % len(base), tag) is not None:
        return tuple(int(part) / 100 for part in tag.split("_")[1:])
    return None


def asymmetry(values: tuple[float, ...]) -> float:
    """How far the values sit from an arithmetic progression, in value units.

    Zero means evenly spaced, which is exactly the case one stored constant can
    express. Sorted descending first, since feature ids go by rank.
    """
    ordered = sorted(values, reverse=True)
    gaps = [a - b for a, b in zip(ordered, ordered[1:])]
    return float(max(gaps) - min(gaps))


def evaluate(checkpoint: Path, levels: str, values: tuple[float, ...], args) -> tuple[float, float]:
    config = MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        n_objectives=len(values),
        objective_values=values,
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )
    policy, _, _, state, _ = load_train_state(checkpoint, env_cfg=config)
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    envs = config.make()
    carry = policy.apply(state.params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
    holder = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

    def act(observations, starts):
        holder["carry"], action, _, holder["key"] = get_action(
            state.params, holder["carry"], observations, starts, holder["key"], temperature=0.0
        )
        return np.asarray(action)

    summary = summarise(collect_episode_outcomes(envs, act, args.episodes, seed=args.seed))
    envs.close()
    return summary.chose_optimal, summary.reached_objective


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}\n")

    base_values = tuple(args.base_values)
    print(f"  {'arm':>16}{'values':>24}{'asymmetry':>11}{'optimal':>10}{'reached':>10}")

    optimal, reached = evaluate(args.base, args.base_levels, base_values, args)
    print(f"  {'base':>16}{str(base_values):>24}{asymmetry(base_values):>11.2f}{optimal:>10.1%}{reached:>10.1%}")

    rows = []
    for run in sorted(args.arms.iterdir()):
        values = values_of(run.name, base_values)
        if values is None:
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*")) if (run / "local-files").is_dir() else []
        levels = args.levels / run.name
        if not checkpoints or not levels.is_dir():
            continue
        optimal, reached = evaluate(checkpoints[-1], str(levels), values, args)
        print(f"  {run.name:>16}{str(values):>24}{asymmetry(values):>11.2f}{optimal:>10.1%}{reached:>10.1%}")
        rows.append((asymmetry(values), optimal))

    if len(rows) >= 4:
        skew = np.array([r[0] for r in rows])
        score = np.array([r[1] for r in rows])
        slope, intercept = np.polyfit(skew, score, 1)
        print(f"\n  optimal against asymmetry: slope {slope:+.3f} per unit, r {np.corrcoef(skew, score)[0, 1]:+.3f}")
        print(f"  fitted at asymmetry 0: {intercept:.1%}   at 0.4: {intercept + 0.4 * slope:.1%}")
        print(
            "\n  A clearly negative slope is the projection account confirmed without touching\n"
            "  weight space: the further an arm's task is from something one stored constant\n"
            "  can express, the worse it does on it. A flat line says the arms hold the\n"
            "  second degree of freedom and the collinear axes are a measurement artefact."
        )


if __name__ == "__main__":
    main()
