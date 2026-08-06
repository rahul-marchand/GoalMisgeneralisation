"""Steer what the actor actually reads, in the step it reads it.

    uv run python experiments/008_steer_at_the_head.py CHECKPOINT --levels DIR

The first attempt steered the recurrent carry between environment steps and
found nothing. The diagnostic said why: one forward pass erased the
displacement completely, 25 cells down to 0.08. The maze is fully observed, so
the recurrence carries computation rather than information, and the next step
simply recomputes the state from the observation. No direction fixes that.

Two changes, both about *where* rather than how hard.

**Steer between the tower and the head.** ``get_action`` is

    carry, hidden = network.step(carry, obs, starts)
    logits, _ = actor(hidden)

so the intervention goes on ``carry[-1].h`` — after the recurrence has run and
before anything reads it. Nothing overwrites it before the action is produced.

**Steer only the last layer.** ``_apply_cells`` returns ``carry[-1].h``, so the
actor sees *one* recurrent layer. A direction spread over all three aimed two
thirds of itself at components the policy never reads.

The direction is a difference of means rather than probe weights — mean
activation where the objective is far minus where it is near — because ridge
returns the minimum-norm direction that predicts, which may point somewhere the
network never travels. The probe is kept only to calibrate it in cells.

Prediction unchanged: displace an objective's field by ``alpha`` and the agent
should behave as though it were ``alpha`` cells further, moving the indifference
point by ``-alpha``. The slope is the result.
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

from goalmisgen.analysis import collect_episode_outcomes, collect_rollouts, fields, geometry, steering, targets
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.probes import Feature, apply_linear, fit_ridge, layer_slice
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33

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
    parser.add_argument("--fit-episodes", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-8, -4, 0, 4, 8])
    parser.add_argument(
        "--differential",
        action="store_true",
        help="Push the two objectives' fields apart instead of shifting one. A uniform offset to a "
        "single field may be invisible to a comparison, the way it is invisible to gradient descent; "
        "moving the two in opposite directions changes the contrast between them by 2*alpha.",
    )
    parser.add_argument(
        "--at-objectives",
        action="store_true",
        help="Apply the shift only at the two objective cells rather than everywhere, in case the "
        "comparison reads specific locations and a global offset washes out.",
    )
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# Flax lets `method=` take a callable receiving the module, so the forward pass
# can be split without subclassing anything or touching third_party. This
# mirrors BaseLSTM.step exactly, including the skip connection: with
# skip_final the readout is `carry[-1].h + embedded`, and reconstructing it
# from the carry alone silently removes a residual path the policy depends on.
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

    # ------------------------------------------------------- probe, last layer
    whole = Feature("activations", operator.attrgetter("features"))
    last = layer_slice(whole, LAST_LAYER, N_LAYERS)

    def probe_for(feature_id: int, probe_params=None):
        target = targets.DistanceToObjective(targets.fixed(feature_id), name=f"d->f{feature_id}", n_features=N_FEATURES)
        rollouts = collect_rollouts(
            env_config(0, args.fit_split).make(), policy, params, args.fit_episodes, seed=0, probe_params=probe_params
        )
        data = fields.cell_data(rollouts, last, target, drop_degenerate=False)
        weights, mean, std = fit_ridge(data.x, data.y, l2=fields.choose_l2(data)[0])
        return data, weights, mean, std

    data, weights, mean, std = probe_for(0)
    decoded = apply_linear(data.x, weights, mean, std)
    quality = float(np.corrcoef(decoded, data.y)[0, 1])
    print(f"last-layer probe on {len(data.y):,} cells, correlation {quality:.3f}")
    if quality < 0.4:
        raise RuntimeError(f"the last layer alone barely decodes the field ({quality:.3f}); steering it is pointless")

    # Difference of means between the cells furthest from and nearest to the
    # objective — a direction the network is observed to travel along.
    far, near = data.y >= np.quantile(data.y, 0.75), data.y <= np.quantile(data.y, 0.25)
    contrast = steering.from_contrast("contrast d->f0", data.x[far], data.x[near], weights, std)
    probe = steering.from_probe("probe d->f0", weights, std)
    cosine = float(probe.delta @ contrast.delta / (probe.unit_norm * contrast.unit_norm))
    print(f"contrast vs probe direction: cosine {cosine:.3f}, norms {contrast.unit_norm:.4f} / {probe.unit_norm:.4f}\n")

    other_data, other_weights, other_mean, other_std = probe_for(1)
    other_far = other_data.y >= np.quantile(other_data.y, 0.75)
    other_near = other_data.y <= np.quantile(other_data.y, 0.25)
    other = steering.from_contrast(
        "contrast d->f1", other_data.x[other_far], other_data.x[other_near], other_weights, other_std
    )

    untrained_params = maze_drc33(min_size=args.size, max_size=args.size).net.init_params(
        env_config(0, args.fit_split).make(), jax.random.PRNGKey(12345)
    )[2]
    u_data, u_weights, _, u_std = probe_for(0, untrained_params)
    u_far, u_near = u_data.y >= np.quantile(u_data.y, 0.75), u_data.y <= np.quantile(u_data.y, 0.25)

    # Pushing the fields apart: objective 0 further, objective 1 nearer, so the
    # difference the comparison would read moves by twice alpha.
    apart = steering.Direction("differential (f0 up, f1 down)", contrast.delta - other.delta)

    # The two directions are not orthogonal, so the cross-effects have to be
    # measured rather than assumed. Steering one field always moves the other a
    # little; if that leak is large the differential arm is not doing what its
    # name says.
    print(f"{'direction':>18}{'moves d->f0':>13}{'moves d->f1':>13}")
    for label, built in (("contrast f0", contrast), ("contrast f1", other), ("differential", apart)):
        on_zero = built.scaled(1.0)[None, :]
        moved_0 = float(apply_linear(on_zero, weights, mean, std) - apply_linear(np.zeros_like(on_zero), weights, mean, std))
        moved_1 = float(
            apply_linear(on_zero, other_weights, other_mean, other_std)
            - apply_linear(np.zeros_like(on_zero), other_weights, other_mean, other_std)
        )
        print(f"{label:>18}{moved_0:>13.3f}{moved_1:>13.3f}")
    print()

    directions = [
        ("differential", apart),
        ("contrast (d->f0)", contrast),
        ("probe (d->f0)", steering.matched("probe (d->f0)", probe, contrast)),
        ("random", steering.matched_random("random", contrast, seed=args.seed)),
        ("other objective", steering.matched("other objective", other, contrast)),
        (
            "untrained",
            steering.matched(
                "untrained",
                steering.from_contrast("u", u_data.x[u_far], u_data.x[u_near], u_weights, u_std),
                contrast,
            ),
        ),
    ]

    # ------------------------------------------------------------- steering
    # One jitted function for the whole split forward pass. Calling the three
    # pieces separately and unjitted costs full tracing overhead on every
    # environment step, which is two orders of magnitude slower and looks like
    # a hung job rather than a slow one. `delta` is an array argument, not a
    # closure, so changing alpha does not force a recompile.
    @jax.jit
    def steered_action(carry, observations, starts, delta, where):
        """``where`` is a (batch, h, w, 1) mask: which cells the shift lands on."""
        carry, readout = policy.apply(params, carry, observations, starts, method=_tower)
        hidden = policy.apply(params, readout + delta * where, method=_mlp)
        logits, _ = policy.apply(params, hidden, method=_actor)
        return carry, jnp.argmax(logits, axis=1)


    def objective_mask(observations) -> np.ndarray:
        """Ones at the two objectives' cells, zero elsewhere.

        Built from the feature channels of the batched NCHW observation, so it
        follows the objectives rather than assuming a fixed position.
        """
        obs = np.asarray(observations)
        marks = obs[:, geometry.FIRST_FEATURE_CHANNEL : geometry.FIRST_FEATURE_CHANNEL + N_FEATURES]
        return (marks.max(axis=1) > 0.5).astype(np.float32)[..., None]



    def measure(direction, alpha: float):

        envs = env_config(args.seed, args.split).make()
        key = jax.random.PRNGKey(args.seed)
        state = {"carry": policy.apply(params, key, envs.observation_space.shape, method=policy.initialize_carry)}


        depth = len(contrast.delta)
        delta = jnp.zeros(depth) if direction is None or alpha == 0 else jnp.asarray(direction.scaled(alpha))

        def act(observations, starts):
            where = jnp.asarray(objective_mask(observations)) if args.at_objectives else jnp.float32(1.0)
            state["carry"], action = steered_action(state["carry"], observations, starts, delta, where)
            return np.asarray(action)

        outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
        gaps, took_richer, _ = value_distance_decisions(outcomes)
        reached = float(np.mean([bool(o.get("reached_objective")) for o in outcomes]))
        return indifference_point(gaps, took_richer), float(took_richer.mean()), reached

    baseline, took, reached = measure(None, 0.0)
    print(f"unsteered indifference {baseline:.2f} extra steps  (took richer {took:.1%}, reached {reached:.1%})")
    # A working agent reaches an objective essentially always. Anything less
    # means the hand-assembled forward pass is not the policy, and every number
    # below it would be a measurement of the reassembly rather than the agent.
    if reached < 0.9:
        raise RuntimeError(
            f"unsteered agent reached an objective in only {reached:.1%} of episodes; the rebuilt forward "
            "pass is not the policy, so no steering number here would mean anything"
        )
    print("baseline reproduces experiment 006, so the split forward pass is the agent\n")

    header = f"{'direction':>18}{'alpha':>7}{'indifference':>14}{'shift':>8}{'took richer':>13}{'reached':>9}"
    print(header)
    for name, direction in directions:
        points = []
        for alpha in args.alphas:
            point, took, reached = measure(direction, alpha)
            points.append((alpha, point))
            print(f"{name:>18}{alpha:>7.0f}{point:>14.2f}{point - baseline:>8.2f}{took:>13.1%}{reached:>9.1%}")
        xs = np.array([a for a, _ in points], dtype=float)
        ys = np.array([p - baseline for _, p in points], dtype=float)
        usable = np.isfinite(ys)
        slope = float(np.polyfit(xs[usable], ys[usable], 1)[0]) if usable.sum() > 1 else float("nan")
        print(f"{'':>18}slope {slope:+.3f}\n")

    print(
        "A slope of -1 means the field is the compared quantity. The random and untrained\n"
        "directions must stay flat, and the other objective's must carry the opposite sign."
    )


if __name__ == "__main__":
    main()
