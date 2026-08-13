"""Is the distance field read, or merely present?

    uv run python experiments/007_steer_distance.py CHECKPOINT --levels DIR

Everything so far is correlational. The recurrent state linearly encodes each
cell's distance to each objective, the field is sharper for the more valuable
objective, and the agent abandons the richer one at 7.8 extra steps. None of
that shows the network *uses* any of it.

The probe hands over a calibrated direction: the smallest activation change
that moves the decoded distance to an objective by one cell. Add ``alpha`` of
it throughout a rollout and there is a quantitative prediction — the agent
should behave as though that objective were ``alpha`` cells further away, so
its indifference point should move by ``-alpha``.

**The slope is the result, not the flip.** Fitting the shift against alpha:

    slope ~ -1   the field is the quantity being compared
    slope ~ -0.3 the field is one input among others
    slope ~ 0    the field is decodable and ignored, and this project needs
                 a different target

Three controls, because a large enough perturbation disturbs any network:

``random``      a direction of identical magnitude pointing nowhere in
                particular; must do nothing
``other``       the *other* objective's field direction; must move the
                threshold the opposite way, which is far harder to explain
                away than a single effect
``untrained``   a direction fitted on an untrained network of the same shape,
                rescaled to the same magnitude

And two schedules. Steering at every step holds the field displaced; steering
once at the episode's start does not. If the one-shot washes out, the field is
continuously re-derived from the observation rather than carried.
"""

from __future__ import annotations

import argparse
import operator
from functools import partial
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen import provenance
from goalmisgen.analysis import collect_episode_outcomes, collect_rollouts, fields, steering, targets
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.probes import Feature, fit_ridge
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33

N_FEATURES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--fit-split", type=str, default="valid", help="Probes are fitted here...")
    parser.add_argument("--split", type=str, default="test", help="...and steering is measured here.")
    parser.add_argument("--fit-episodes", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=2048, help="Per steering condition.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0, help="At 1.0 feature 0 is always the richer one.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[-6, -3, 0, 3, 6])
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(provenance.header())

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
    print(f"checkpoint {args.checkpoint.name}  (update {update})\n")

    _, _, untrained_params = maze_drc33(min_size=args.size, max_size=args.size).net.init_params(
        env_config(0, args.fit_split).make(), jax.random.PRNGKey(12345)
    )

    # ---------------------------------------------------------------- probes
    activations = Feature("activations", operator.attrgetter("features"))

    def direction_for(feature_id: int, probe_params=None) -> steering.Direction:
        target = targets.DistanceToObjective(targets.fixed(feature_id), name=f"d->f{feature_id}", n_features=N_FEATURES)
        rollouts = collect_rollouts(
            env_config(0, args.fit_split).make(),
            policy,
            train_state.params,
            args.fit_episodes,
            seed=0,
            probe_params=probe_params,
        )
        # Degenerate columns must stay: a direction has to span the whole hidden
        # state or it cannot be added back to it.
        data = fields.cell_data(rollouts, activations, target, drop_degenerate=False)
        weights, mean, std = fit_ridge(data.x, data.y, l2=fields.choose_l2(data)[0])
        built = steering.from_probe(f"d->f{feature_id}", weights, std)

        achieved = steering.verify(built, weights, mean, std, 3.0)
        if abs(achieved - 3.0) > 1e-4:
            raise RuntimeError(f"steering is miscalibrated: asking for 3 cells moved the decode by {achieved:.4f}")
        return built

    richer = direction_for(0)
    poorer = direction_for(1)
    print(f"steering 1 cell costs an activation change of norm {richer.unit_norm:.4f}")

    controls = [
        steering.matched_random("random", richer, seed=args.seed),
        steering.matched("other objective", poorer, richer),
        steering.matched("untrained", direction_for(0, untrained_params), richer),
    ]

    # ---------------------------------------------------------------- steering
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    def measure(direction: steering.Direction | None, alpha: float, persistent: bool) -> tuple[float, float, float]:
        envs = env_config(args.seed, args.split).make()
        key = jax.random.PRNGKey(args.seed)
        state = {
            "carry": policy.apply(train_state.params, key, envs.observation_space.shape, method=policy.initialize_carry),
            "key": key,
        }
        delta = None if direction is None or alpha == 0 else direction.scaled(alpha)

        def act(observations, starts):
            state["carry"], action, _, state["key"] = get_action(
                train_state.params, state["carry"], observations, starts, state["key"], temperature=0.0
            )
            # After the forward pass, so the displacement is carried into the
            # next decision. On a start step get_action has just cleared the
            # carry, which is why the one-shot schedule steers exactly there.
            if delta is not None and (persistent or bool(np.asarray(starts).any())):
                state["carry"] = steering.apply_to_carry(state["carry"], delta)
            return np.asarray(action)

        outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
        gaps, took_richer, _ = value_distance_decisions(outcomes)
        reached = float(np.mean([bool(o.get("reached_objective")) for o in outcomes]))
        return indifference_point(gaps, took_richer), float(took_richer.mean()), reached

    baseline, _, _ = measure(None, 0.0, True)
    print(f"unsteered indifference {baseline:.2f} extra steps\n")

    header = f"{'schedule':>12}{'direction':>18}{'alpha':>7}{'indifference':>14}{'shift':>8}{'took richer':>13}{'reached':>9}"
    print(header)
    slopes = {}
    for persistent in (True, False):
        schedule = "every step" if persistent else "once"
        points = []
        for alpha in args.alphas:
            point, took, reached = measure(richer, alpha, persistent)
            points.append((alpha, point))
            print(
                f"{schedule:>12}{'d->f0 (richer)':>18}{alpha:>7.0f}{point:>14.2f}"
                f"{point - baseline:>8.2f}{took:>13.1%}{reached:>9.1%}"
            )
        alphas = np.array([a for a, _ in points], dtype=float)
        shifts = np.array([p - baseline for _, p in points], dtype=float)
        usable = np.isfinite(shifts)
        slopes[schedule] = float(np.polyfit(alphas[usable], shifts[usable], 1)[0]) if usable.sum() > 1 else float("nan")
        print(f"{'':>12}slope {slopes[schedule]:+.3f} cells of threshold per cell of steering\n")

    print(f"{'':>12}{'CONTROLS (every step)':>18}")
    for control in controls:
        for alpha in (min(args.alphas), max(args.alphas)):
            point, took, reached = measure(control, alpha, True)
            print(
                f"{'every step':>12}{control.name:>18}{alpha:>7.0f}{point:>14.2f}"
                f"{point - baseline:>8.2f}{took:>13.1%}{reached:>9.1%}"
            )

    print(
        f"\nA slope of -1 would mean the field is the compared quantity. Measured "
        f"{slopes.get('every step', float('nan')):+.3f} when steered at every step and "
        f"{slopes.get('once', float('nan')):+.3f} when steered once.\n"
        "The random control must be flat and the other-objective control must have the opposite sign;\n"
        "if either fails, the perturbation is disturbing the network rather than steering a quantity."
    )


if __name__ == "__main__":
    main()
