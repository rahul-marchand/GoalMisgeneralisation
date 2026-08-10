"""Is an agent worth running an experiment around?

    uv run python scripts/competence.py CHECKPOINT --levels DIR [--min-reached 95]

Exits non-zero when the agent fails the bar, so a long unattended script can
stop rather than spend hours sweeping around an agent whose numbers would mean
nothing.

The bar is deliberately about *reaching* rather than choosing well. An agent
that reaches an objective every time but chooses the wrong one has a goal worth
studying; an agent that wanders until the step limit has no goal to study, and
every downstream measurement — exchange rates, weight diffs — is then reading
noise. Choice quality is printed alongside for the log, not gated on.

Parse failures are not treated as failure. This measures the agent itself
rather than scraping another script's output, precisely so that "I could not
tell" cannot be confused with "the agent is bad".
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-objectives", type=int, default=3)
    parser.add_argument("--objective-values", type=float, nargs="+", required=True)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-reached", type=float, default=95.0, help="Percent, below which this exits non-zero.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        n_objectives=args.n_objectives,
        objective_values=tuple(args.objective_values),
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split=args.split,
        asynchronous=False,
        seed=args.seed,
    )

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=config)
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    envs = config.make()
    carry = policy.apply(
        train_state.params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry
    )
    state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

    def act(observations, starts):
        state["carry"], action, _, state["key"] = get_action(
            train_state.params, state["carry"], observations, starts, state["key"], temperature=0.0
        )
        return np.asarray(action)

    summary = summarise(collect_episode_outcomes(envs, act, args.episodes, seed=args.seed))
    print(f"checkpoint {args.checkpoint.name}  (update {update})")
    print(f"  reached        {summary.reached_objective:.1%}")
    print(f"  chose optimal  {summary.chose_optimal:.1%}")
    print(f"  followed f0    {summary.followed_feature_zero:.1%}")
    print(f"  ambiguous      {summary.ambiguous:.1%}")
    print(f"  return / steps {summary.mean_return:.3f} / {summary.mean_steps:.1f}")

    if summary.reached_objective * 100 < args.min_reached:
        print(f"\nBELOW BAR: reached {summary.reached_objective:.1%} against a required {args.min_reached:g}%")
        sys.exit(1)
    print(f"\nfit to sweep around: reached {summary.reached_objective:.1%}")


if __name__ == "__main__":
    main()
