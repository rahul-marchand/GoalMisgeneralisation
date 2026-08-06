"""Can steering move this agent's choice at all?

    uv run python experiments/010_choice_direction.py CHECKPOINT --levels DIR

Four controlled attempts have failed to shift the objective the agent takes by
displacing its distance field. Every one of them has the same alternative
explanation, and it is not about distance: **steering may be inert in this
architecture.** If no direction can move the choice, those nulls say nothing
about whether the field is used.

This is the control that decides it, and it is deliberately the easiest
possible case. The direction is built from the outcome itself — mean activation
on episodes where the agent took objective 0, minus episodes where it took
objective 1 — matched on the distance gap so the contrast is about the choice
rather than about the levels. If a direction derived from the behaviour cannot
move the behaviour, nothing will, and the four nulls are a fact about the
method rather than about the network.

Registered before running:

* if the choice direction shifts the indifference point and the norm-matched
  controls stay flat, steering works here and the distance nulls are real
* if it too is flat, every steering result so far is uninterpretable and the
  causal question needs a different tool entirely — patching, or an
  intervention inside the tick loop

The fit is on disjoint episodes from the measurement, so "a direction built
from behaviour predicts behaviour" cannot be circular.
"""

from __future__ import annotations

import argparse
import operator
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import collect_episode_outcomes, collect_rollouts, geometry, steering
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.probes import Feature, layer_slice
from goalmisgen.configs.env import MazeConfig

N_FEATURES = 2
N_LAYERS = 3
LAST_LAYER = N_LAYERS - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--fit-split", type=str, default="valid")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit-episodes", type=int, default=1024)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--scales", type=float, nargs="+", default=[-3, -1.5, 0, 1.5, 3])
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _tower(module, carry, obs, starts):
    net = module.network_params
    embedded = net._compress_input(module._maybe_normalize_input_image(obs))
    out_carry, readout = net._apply_cells(carry, embedded, starts)
    if net.cfg.skip_final:
        readout = readout + embedded
    return out_carry, readout


def _mlp(module, hidden):
    return module.network_params._mlp(hidden)


def _actor(module, hidden):
    return module.actor_params(hidden)


def choice_contrast(rollouts, feature: Feature, gap_tolerance: float = 6.0):
    """Mean activation on 'took objective 0' minus 'took objective 1'.

    Matched on the distance gap: without that the contrast is dominated by the
    levels the two groups came from, since the agent takes the nearer objective
    far more often. Pairing episodes with similar gaps leaves the choice as the
    thing that differs.
    """
    took, gaps, vectors = [], [], []
    for rollout in rollouts:
        reached = rollout.info.get("reached_feature_id")
        if reached is None:
            continue
        first = rollout.info.get("feature_0_distance")
        second = rollout.info.get("feature_1_distance")
        if first is None or second is None or first < 0 or second < 0:
            continue
        row, col = geometry.agent_cell(rollout.observation)
        took.append(int(reached))
        gaps.append(float(first - second))
        vectors.append(feature(rollout)[row, col])

    took = np.array(took)
    gaps = np.array(gaps)
    vectors = np.stack(vectors)

    # Pair each episode with the nearest-gap episode of the other choice, so the
    # difference is taken between comparable levels.
    zero, one = np.flatnonzero(took == 0), np.flatnonzero(took == 1)
    if min(len(zero), len(one)) < 30:
        raise RuntimeError(f"only {len(zero)} / {len(one)} episodes per choice; the contrast would be noise")

    pairs = []
    for index in zero:
        candidate = one[np.argmin(np.abs(gaps[one] - gaps[index]))]
        if abs(gaps[candidate] - gaps[index]) <= gap_tolerance:
            pairs.append((index, candidate))
    if len(pairs) < 30:
        raise RuntimeError(f"only {len(pairs)} matched pairs within {gap_tolerance} cells")

    difference = np.mean([vectors[a] - vectors[b] for a, b in pairs], axis=0)
    return difference, len(pairs)


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}")

    def env_config(seed: int, split: str) -> MazeConfig:
        settings: dict[str, object] = dict(
            max_episode_steps=120,
            num_envs=args.num_envs,
            min_size=args.size,
            max_size=args.size,
            feature_value_correlation=args.correlation,
            randomise_values=args.randomise_values,
            level_dataset=args.levels,
            asynchronous=False,
            seed=seed,
        )
        if args.levels:
            settings["dataset_split"] = split
        return MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=env_config(0, args.fit_split))
    params = train_state.params
    print(f"checkpoint {args.checkpoint.name}  (update {update})\n")

    last = layer_slice(Feature("activations", operator.attrgetter("features")), LAST_LAYER, N_LAYERS)
    fit_rollouts = collect_rollouts(
        env_config(0, args.fit_split).make(), policy, params, args.fit_episodes, seed=0
    )
    raw, pairs = choice_contrast(fit_rollouts, last)
    choice = steering.Direction("choice (f0 - f1)", raw / np.linalg.norm(raw))
    print(f"choice direction from {pairs} gap-matched pairs, norm {np.linalg.norm(raw):.4f}")
    print("scales are multiples of that unit direction, not cells — this quantity has no natural unit\n")

    controls = [
        ("random", steering.matched_random("random", choice, seed=args.seed)),
    ]

    @jax.jit
    def steered_action(carry, observations, starts, delta):
        carry, readout = policy.apply(params, carry, observations, starts, method=_tower)
        hidden = policy.apply(params, readout + delta, method=_mlp)
        logits, _ = policy.apply(params, hidden, method=_actor)
        return carry, jnp.argmax(logits, axis=1)

    def measure(direction, scale: float):
        envs = env_config(args.seed, args.split).make()
        key = jax.random.PRNGKey(args.seed)
        state = {"carry": policy.apply(params, key, envs.observation_space.shape, method=policy.initialize_carry)}
        depth = len(choice.delta)
        delta = jnp.zeros(depth) if direction is None or scale == 0 else jnp.asarray(direction.scaled(scale))

        def act(observations, starts):
            state["carry"], action = steered_action(state["carry"], observations, starts, delta)
            return np.asarray(action)

        outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
        gaps, took_richer, _ = value_distance_decisions(outcomes)
        reached = float(np.mean([bool(o.get("reached_objective")) for o in outcomes]))
        took_zero = float(np.mean([o.get("reached_feature_id") == 0 for o in outcomes if o.get("reached_objective")]))
        return indifference_point(gaps, took_richer), took_zero, reached

    baseline, took_zero, reached = measure(None, 0.0)
    print(f"unsteered indifference {baseline:.2f}  (took feature 0 {took_zero:.1%}, reached {reached:.1%})")
    if reached < 0.9:
        raise RuntimeError(f"unsteered agent reached only {reached:.1%}; the forward pass is not the policy")
    print()

    print(f"{'direction':>14}{'scale':>7}{'indifference':>14}{'shift':>8}{'took f0':>10}{'reached':>9}")
    for name, direction in [("choice", choice), *controls]:
        for scale in args.scales:
            point, took_zero, reached = measure(direction, scale)
            print(f"{name:>14}{scale:>7.1f}{point:>14.2f}{point - baseline:>8.2f}{took_zero:>10.1%}{reached:>9.1%}")
        print()

    print(
        "If the choice direction moves the threshold and random does not, steering works here\n"
        "and the four distance nulls are about the field rather than the method. If both are\n"
        "flat, no steering result so far is interpretable."
    )


if __name__ == "__main__":
    main()
