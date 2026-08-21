"""Do the enriched channels carry the decision, in the agent's own activations?

    uv run python experiments/020_are_they_value_channels.py \
        --base /workspace/data/runs/novalue11.s1234/local-files/cp_140206080 \
        --channels 7 1 --layer 0 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --objective-values 1.0 0.5

Everything so far is about weights. ``018`` found two channels of the first
recurrent layer carrying about twice their share of the value axis, and ``019``
found that masking to them makes the trained offset readable off a held-out
checkpoint while costing most of the ability to write it. All of that describes
where fine-tuning *wrote*. None of it says the channels represent anything.

Three tests here, none of which involve a fitted axis, so none inherit its noise.

``ablate``   zero those channels in the untouched agent, every step, and see
             whether the exchange rate collapses toward distance-only while the
             agent still reaches objectives. Scored against random channel pairs
             and against the channels the axis avoids, so "any two channels
             matter" and "these two matter" are told apart.
``probe``    rank all 32 channels by how well their activation before the first
             move predicts which objective the agent will walk to. An
             independent ranking that reproduces the weight-space one would be
             two unrelated methods agreeing.
``steer``    add to those channels during the rollout and see whether the
             exchange rate moves, against norm-matched random channels. The
             activation analogue of the weight edit, and it may work where that
             one was limited, since it does not need the change to be written
             into a distributed set of weights.

The probe reads at the agent's own position. Values are constants for this
agent, so there is no per-episode value to regress against; what varies is which
objective wins, and that is what a value comparison would have to produce.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen import provenance
from goalmisgen.analysis import collect_episode_outcomes, summarise
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.probes import roc_auc
from goalmisgen.configs.env import MazeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--channels", type=int, nargs="+", default=[7, 1])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--objective-values", type=float, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controls", type=int, default=4, help="Random channel pairs to score against.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[-3.0, -1.0, 1.0, 3.0])
    parser.add_argument("--skip", type=str, nargs="*", default=[], choices=["ablate", "probe", "steer"])
    return parser.parse_args()


def config_for(args) -> MazeConfig:
    return MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        n_objectives=len(args.objective_values),
        objective_values=tuple(args.objective_values),
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )


def edit_layer(carry, layer: int, channels, mode: str, amount: float = 0.0):
    """Zero or shift chosen channels of one layer, in both h and c.

    Both, because they are recomputed on different schedules: ``h`` is rebuilt
    from the gates on the next tick, so editing it alone is undone almost
    immediately, while ``c`` is the layer's persistent memory. An ablation that
    only touched ``h`` would understate the channel's importance for reasons
    that have nothing to do with what it represents.
    """
    edited = []
    for index, cell in enumerate(carry):
        if index != layer:
            edited.append(cell)
            continue
        h, c = np.asarray(cell.h), np.asarray(cell.c)
        h, c = h.copy(), c.copy()
        for channel in channels:
            if mode == "zero":
                h[..., channel] = 0.0
                c[..., channel] = 0.0
            else:
                h[..., channel] += amount
                c[..., channel] += amount
        edited.append(cell.replace(h=jax.numpy.asarray(h), c=jax.numpy.asarray(c)))
    return edited


def run(params, policy, envs, args, label, transform=None):
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    carry = policy.apply(params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
    state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

    def act(observations, starts):
        state["carry"], action, _, state["key"] = get_action(
            params, state["carry"], observations, starts, state["key"], temperature=0.0
        )
        if transform is not None:
            state["carry"] = transform(state["carry"])
        return np.asarray(action)

    outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
    gaps, took, _ = value_distance_decisions(outcomes)
    point = indifference_point(gaps, took)
    summary = summarise(outcomes)
    print(f"  {label:>40}{point:>8.1f}   optimal {summary.chose_optimal:>6.1%}   reached {summary.reached_objective:>6.1%}")
    return point, summary


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")

    config = config_for(args)
    policy, _, _, state, _ = load_train_state(args.base, env_cfg=config)
    params = state.params
    envs = config.make()
    rng = np.random.default_rng(args.seed)
    target = list(args.channels)
    print(f"layer {args.layer} of the recurrent stack, channels {target}\n")

    if "ablate" not in args.skip:
        print("=== ablation: zero those channels in the untouched agent ===\n")
        print(f"  {'':>40}{'steps':>8}")
        run(params, policy, envs, args, "untouched")
        run(
            params,
            policy,
            envs,
            args,
            f"channels {target} zeroed",
            partial(edit_layer, layer=args.layer, channels=target, mode="zero"),
        )
        for trial in range(args.controls):
            pick = list(rng.choice([c for c in range(32) if c not in target], size=len(target), replace=False))
            run(
                params,
                policy,
                envs,
                args,
                f"random pair {pick} zeroed",
                partial(edit_layer, layer=args.layer, channels=pick, mode="zero"),
            )
        print(
            "\n  A value comparison removed should push the exchange rate toward zero -- the\n"
            "  agent taking whichever objective is nearest -- while it still reaches one.\n"
            "  Random pairs say how much of any effect is just losing two channels."
        )

    if "steer" not in args.skip:
        print("\n\n=== steering: add to those channels during the rollout ===\n")
        print(f"  {'':>40}{'steps':>8}")
        run(params, policy, envs, args, "untouched")
        for alpha in args.alphas:
            run(
                params,
                policy,
                envs,
                args,
                f"channels {target} {alpha:+.1f}",
                partial(edit_layer, layer=args.layer, channels=target, mode="shift", amount=alpha),
            )
        pick = list(rng.choice([c for c in range(32) if c not in target], size=len(target), replace=False))
        for alpha in (min(args.alphas), max(args.alphas)):
            run(
                params,
                policy,
                envs,
                args,
                f"random pair {pick} {alpha:+.1f}",
                partial(edit_layer, layer=args.layer, channels=pick, mode="shift", amount=alpha),
            )
        print(
            "\n  A channel holding what an objective is worth should move the exchange rate\n"
            "  monotonically in the amount added, and the random pair should not."
        )

    if "probe" not in args.skip:
        print("\n\n=== probe: which channels predict the objective the agent takes? ===\n")
        get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
        activations, labels = [], []
        batches = -(-args.episodes // args.num_envs)
        for batch in range(batches):
            observations, _ = envs.reset(seed=args.seed + 10_000 + batch)
            carry = policy.apply(
                params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry
            )
            key = jax.random.PRNGKey(args.seed)
            starts = np.ones(envs.num_envs, dtype=bool)
            carry, action, _, key = get_action(params, carry, observations, starts, key, temperature=0.0)

            # Pool over free cells: the decision is an episode-level fact, and a
            # per-cell readout would need a position to read at that is itself
            # chosen by the answer.
            layer = carry[args.layer]
            activations.append(np.asarray(layer.h).mean(axis=(1, 2)))

            done = np.zeros(envs.num_envs, dtype=bool)
            outcome = [None] * envs.num_envs
            for _ in range(config.max_episode_steps + 1):
                observations, _, terminated, truncated, info = envs.step(np.asarray(action))
                finished = np.logical_or(terminated, truncated)
                finals = info.get("final_info")
                for index, was_done in enumerate(finished):
                    if was_done and not done[index] and finals is not None and finals[index] is not None:
                        outcome[index] = dict(finals[index])
                        done[index] = True
                if done.all():
                    break
                carry, action, _, key = get_action(params, carry, observations, finished, key, temperature=0.0)
            labels.append(outcome)

        rows = [(a, o) for batch_a, batch_o in zip(activations, labels) for a, o in zip(batch_a, batch_o) if o is not None]
        keep = [
            (a, float(o.get("reached_feature_id") == 0))
            for a, o in rows
            if o.get("reached_objective") and o.get("feature_0_value") != o.get("feature_1_value")
        ]
        if len(keep) < 100:
            print(f"  only {len(keep)} usable episodes, skipping")
            return
        features = np.stack([a for a, _ in keep])
        took_zero = np.array([label for _, label in keep])
        print(f"  {len(keep):,} episodes, {took_zero.mean():.1%} took feature 0\n")

        scores = np.array([abs(roc_auc(took_zero, features[:, channel]) - 0.5) for channel in range(features.shape[1])])
        order = np.argsort(scores)[::-1]
        print("  channels ranked by how well their activation predicts the choice:")
        print("    " + "  ".join(f"ch{c:02d}({0.5 + scores[c]:.3f})" for c in order[:8]))
        print(
            f"\n  the weight-space picks {target} sit at ranks "
            + ", ".join(str(int(np.where(order == c)[0][0]) + 1) for c in target)
        )
        print(
            "\n  An independent ranking that puts the same channels on top would be two\n"
            "  unrelated methods agreeing. One that does not means the axis wrote where it\n"
            "  was cheap to write, which is not where the decision is held."
        )


if __name__ == "__main__":
    main()
