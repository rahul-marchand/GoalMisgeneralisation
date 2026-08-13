"""What exchange rate between value and distance did the agent actually learn?

    uv run python experiments/006_psychometric.py CHECKPOINT --levels DIR

The task defines utility as ``value - 0.05 x distance``. With objectives worth
1.0 and 0.5, an optimal agent abandons the richer one exactly when it is **10
steps** further away, because that is where the 0.5 value gap is cancelled by
the walk. That number is never shown to the agent — the step penalty arrives
only through reward — so whatever rate it learned is internal, and measurable.

This measures it. For every episode, how much further away the richer objective
was, and whether the agent took it anyway. The point where that curve crosses
50% is the agent's indifference distance, and ``value gap / indifference`` is
the step penalty it behaves as though it believes.

Why this comes before any intervention: a threshold you have not measured is a
threshold you cannot steer. If the agent's indifference sits at 10 it has
learned the true rate; if it sits elsewhere, the gap is itself a finding and a
far better target, because you know which way you are pushing.

Purely behavioural — no probe, no activations. It says what the agent does, not
where it does it.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen import provenance
from goalmisgen.analysis import collect_episode_outcomes, metrics
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.configs.env import MazeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=4096, help="Each contributes one decision, so be generous.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.05, help="The rate the task actually charges.")
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument(
        "--hide-values",
        action="store_true",
        help="Match a run trained without a value channel; its observations have one channel fewer.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(provenance.header())

    settings: dict[str, object] = dict(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        feature_value_correlation=args.correlation,
        randomise_values=args.randomise_values,
        value_encoding="none" if args.hide_values else "at_objective",
        colour_is_the_only_value_cue=args.hide_values,
        level_dataset=args.levels,
        asynchronous=False,
        seed=args.seed,
    )
    if args.levels:
        settings["dataset_split"] = args.split
    config = MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=config)
    print(f"checkpoint {args.checkpoint.name}  (update {update})\n")

    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    envs = config.make()
    carry = policy.apply(train_state.params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
    state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

    def act(observations, starts):
        state["carry"], action, _, state["key"] = get_action(
            train_state.params, state["carry"], observations, starts, state["key"], temperature=0.0
        )
        return np.asarray(action)

    outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
    gaps, took_richer, value_gaps = value_distance_decisions(outcomes)
    print(f"{len(gaps):,} of {len(outcomes):,} episodes posed a value-versus-distance trade-off\n")

    print(f"{'extra steps to the richer one':>32}{'n':>8}{'took it':>10}")
    edges = [-40, -12, -8, -4, -1, 2, 6, 10, 14, 18, 40]
    for low, high in zip(edges, edges[1:]):
        rows = (gaps >= low) & (gaps < high)
        if rows.sum() < 20:
            continue
        print(f"{f'{low:+d} to {high:+d}':>32}{int(rows.sum()):>8}{took_richer[rows].mean():>10.1%}")

    point = indifference_point(gaps, took_richer)
    low, high = metrics.bootstrap_episodes(
        lambda rows: indifference_point(gaps[rows], took_richer[rows]), np.arange(len(gaps)), resamples=200, seed=args.seed
    )

    expected = float(np.median(value_gaps)) / args.step_penalty
    print(f"\nindifference at {point:.1f} extra steps  [{low:.1f}, {high:.1f}]")
    print(f"the task's own rate puts it at {expected:.1f}")
    if np.isfinite(point) and point > 0:
        print(f"implied step penalty {np.median(value_gaps) / point:.4f} against the true {args.step_penalty:g}")
    print(
        "\nAbove the indifference point the richer objective is not worth the walk.\n"
        "An agent that had learned the task's rate exactly would cross 50% there."
    )


if __name__ == "__main__":
    main()
