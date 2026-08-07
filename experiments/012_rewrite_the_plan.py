"""Rewrite the agent's route in its cell state, and see if it walks the new one.

    uv run python experiments/012_rewrite_the_plan.py CHECKPOINT --levels DIR

This is the planning-interpretability intervention (arXiv 2504.01871), ported.
Five nulls so far have displaced a *scalar* — a distance, a value — by adding one
vector, uniformly, to the hidden state at the tower's output. The control in
``010`` showed the failure was the method: a direction built from the agent's own
choices moved behaviour no more than a random vector of the same norm.

Four things change here, and each is a difference from the paper we had not been
observing:

**The cell state, not the readout.** ``c`` is the layer's persistent memory,
carried across ticks and steps. ``h`` is recomputed from the gates every tick,
and the readout we were editing has no recurrence downstream of it at all — one
Dense layer, then the actor. An edit at the readout gets one linear map to
express itself against a plan that is already formed.

**A plan, not a nudge.** The concept is directional and discrete: five classes
per cell, four moves or ``NEVER``. "Go right from here" is a coherent thing to
write; "this cell is 3 away when its neighbours say 8" describes no maze, so the
network is handed a contradiction and sensibly ignores it.

**Many cells, consistently.** The edit writes a whole alternative route, and
writes ``NEVER`` along the route it is replacing. A uniform constant added at
every cell — what ``apply_to_carry`` does — leaves a field's argmin and gradient
untouched, so a policy reading the field's shape is provably unaffected and that
null was guaranteed before it was run.

**Every step**, since the plan has to survive being overwritten.

Our mazes are perfect — exactly one path between any two cells — so the paper's
"shortcut" intervention has nothing to divert onto. The directional intervention
ports exactly, and lands on the question this project is about: write the route
to the objective the agent is *not* taking, and see whether it goes there.

Registered before running:

* the plan direction switches the objective and the norm-matched controls do
  not → intervention works here, and the five scalar nulls become real findings
  about distance and value rather than about the method
* the plan direction does no better than a shuffled plan → the edit is
  disruption, not steering, and additive intervention is inert in this
  architecture at any site we can reach

Controls, all at identical norms and identical cells:

``random``    fixed random unit vectors in place of the probe's class directions
``shuffled``  a real plan, for a different episode's maze
``self``      the plan for the objective the agent was already going to take

``self`` is the one that separates steering from damage: it perturbs just as
many cells just as hard, and should change nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis import collect_rollouts, geometry, metrics, plans, steering
from goalmisgen.analysis.probes import (
    apply_multinomial,
    class_directions,
    class_write_accuracy,
    fit_multinomial,
)
from goalmisgen.configs.env import MazeConfig

N_FEATURES = 2
N_LAYERS = 3
STEP_PENALTY = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--fit-split", type=str, default="valid")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fit-episodes", type=int, default=512)
    parser.add_argument("--batches", type=int, default=8, help="Episodes measured are batches x num-envs.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
        help="Norm of the written vector, as a multiple of the typical per-cell cell-state norm. "
        "The probe reads the written class from about 0.2 upward and the policy starts losing "
        "episodes around 0.75, so the interpretable band is narrow and worth sampling finely.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2], help="Which layers to write to.")
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Also sweep each layer alone, as the paper reports interventions per layer.",
    )
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument(
        "--hide-values",
        action="store_true",
        help="Match a run trained without a value channel; utilities then come from --feature-values.",
    )
    parser.add_argument(
        "--feature-values",
        type=float,
        nargs="+",
        default=None,
        help="What each feature is worth, when the observation does not carry it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write every arm here, so a figure is drawn from a file this script produced.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Level ground truth, read from the observation
# ----------------------------------------------------------------------


def objective_values(observation: np.ndarray, fixed: list[float] | None) -> list[float]:
    if fixed is not None:
        return list(fixed)
    return [geometry.objective_value(observation, feature, N_FEATURES) for feature in range(N_FEATURES)]


def optimal_objective(observation: np.ndarray, fixed: list[float] | None) -> int | None:
    """Which objective an optimal agent takes. ``None`` on a tie or if one is cut off.

    The intervention's target is the *other* one, so this decides what counts as
    a switch. Ties are dropped rather than broken: an episode where the two are
    worth the same would be a success under either label.
    """
    values = objective_values(observation, fixed)
    utilities = []
    for feature in range(N_FEATURES):
        field = geometry.bfs_field(
            geometry.blocking_walls(observation, feature, N_FEATURES),
            geometry.objective_cell(observation, feature),
        )
        distance = field[geometry.agent_cell(observation)]
        if not np.isfinite(distance):
            return None
        utilities.append(values[feature] - STEP_PENALTY * float(distance))

    best = max(utilities)
    return None if utilities.count(best) > 1 else int(np.argmax(utilities))


def utility_gap(observation: np.ndarray, fixed: list[float] | None) -> float:
    """How much the optimal objective beats the other, in reward.

    The intervention is asking the agent to give this up, so it is the natural
    axis to stratify success along — and it is why handcrafting levels is not
    needed here: the paper's "plausible alternative" case is simply the small-gap
    end of a distribution we already sample.
    """
    values = objective_values(observation, fixed)
    utilities = []
    for feature in range(N_FEATURES):
        field = geometry.bfs_field(
            geometry.blocking_walls(observation, feature, N_FEATURES),
            geometry.objective_cell(observation, feature),
        )
        utilities.append(values[feature] - STEP_PENALTY * float(field[geometry.agent_cell(observation)]))
    return float(abs(utilities[0] - utilities[1]))


# ----------------------------------------------------------------------
# The edit
# ----------------------------------------------------------------------


def plan_edit(
    observation: np.ndarray,
    target: int,
    replaced: int,
    write_route: bool = True,
    erase_old: bool = True,
) -> dict[tuple[int, int], int] | None:
    """Which class to write at which cell: the new route, and NEVER on the old.

    ``None`` if the target cannot be reached. Only cells that carry information
    are touched — the route to write and the route to erase. Writing NEVER at
    every off-route cell instead would perturb most of the maze, which measures
    how much damage the policy tolerates rather than whether it follows a plan.

    The two halves separate because they are different claims. Erasing the old
    route says "not that way" and needs no plan at all; writing the new one says
    "this way". If erasing alone moves the agent, the edit is working as a
    blockade and the directional concept is doing nothing — which would be worth
    knowing before calling any of this a plan.
    """
    edit: dict[tuple[int, int], int] = {}

    if write_route:
        wanted = plans.planned_directions(observation, target, N_FEATURES)
        if wanted is None:
            return None
        for row, col in np.argwhere((wanted >= 0) & (wanted < plans.NEVER)):
            edit[(int(row), int(col))] = int(wanted[row, col])

    if erase_old:
        old = plans.planned_directions(observation, replaced, N_FEATURES)
        if old is None:
            return None if not edit else edit
        for row, col in np.argwhere((old >= 0) & (old < plans.NEVER)):
            # A cell shared by both routes keeps its new direction: the agent
            # walks it either way, so erasing it would contradict the plan we
            # are writing rather than the one we are replacing.
            edit.setdefault((int(row), int(col)), plans.NEVER)
    return edit or None


def build_deltas(
    edits: list[dict[tuple[int, int], int] | None],
    directions: dict[int, np.ndarray],
    shape: tuple[int, int],
    channels: int,
    magnitude: float,
) -> list[jnp.ndarray]:
    """Per-layer ``(n_envs, height, width, channels)`` arrays to add to ``c``."""
    height, width = shape
    grids = [np.zeros((len(edits), height, width, channels), dtype=np.float32) for _ in range(N_LAYERS)]
    for index, edit in enumerate(edits):
        if edit is None:
            continue
        for (row, col), label in edit.items():
            for layer, table in directions.items():
                grids[layer][index, row, col] = magnitude * table[label]
    return [jnp.asarray(grid) for grid in grids]


def derange(count: int, seed: int) -> np.ndarray:
    """A permutation with no fixed point, so no plan lands on its own maze."""
    order = np.random.default_rng(seed).permutation(count)
    for position, destination in enumerate(order):
        if position == destination:
            swap = (position + 1) % count
            order[position], order[swap] = order[swap], order[position]
    return order


# ----------------------------------------------------------------------
# Rollouts under intervention
# ----------------------------------------------------------------------


def measure(
    envs,
    policy,
    params,
    get_action,
    batches: int,
    seed: int,
    directions: dict[int, np.ndarray],
    magnitude: float,
    arm: str,
    fixed_values: list[float] | None,
    channels: int,
) -> list[dict]:
    """Run episodes writing a plan into the cell state before every step.

    The write happens *before* each forward pass, including the first, so the
    plan is present for all nine ticks of every step rather than arriving after
    the action has been chosen. At ``magnitude == 0`` this is exactly the
    unmodified agent, which is what makes the alpha=0 row a real baseline.
    """
    records: list[dict] = []
    key = jax.random.PRNGKey(seed)

    for batch in range(batches):
        observations, _ = envs.reset(seed=seed + batch)
        n_envs = envs.num_envs
        frames = [np.moveaxis(np.asarray(observations)[index], 0, -1) for index in range(n_envs)]

        optimal = [optimal_objective(frame, fixed_values) for frame in frames]
        gaps = [utility_gap(frame, fixed_values) for frame in frames]

        write_route = arm != "erase"
        erase_old = arm not in ("route", "self-route")
        edits: list[dict[tuple[int, int], int] | None] = []
        for frame, best in zip(frames, optimal):
            if best is None:
                edits.append(None)
                continue
            # "self" points the same machinery where the agent was already going.
            pointed_at, replaced = (best, 1 - best) if arm.startswith("self") else (1 - best, best)
            edits.append(plan_edit(frame, pointed_at, replaced, write_route, erase_old))
        if arm == "shuffled":
            edits = [edits[index] for index in derange(n_envs, seed + batch)]

        height, width = frames[0].shape[0], frames[0].shape[1]
        deltas = build_deltas(edits, directions, (height, width), channels, magnitude)

        carry = policy.apply(params, key, envs.observation_space.shape, method=policy.initialize_carry)
        done = np.zeros(n_envs, dtype=bool)
        finals: list[dict | None] = [None] * n_envs
        starts = np.zeros(n_envs, dtype=bool)
        steps_taken = np.zeros(n_envs, dtype=np.int64)

        budget = getattr(envs, "max_episode_steps", None) or 512
        for _ in range(budget + 1):
            if magnitude != 0.0:
                carry = steering.write_to_cell_state(carry, deltas)
            carry, action, _, key = get_action(params, carry, observations, starts, key, temperature=0.0)
            observations, _, terminated, truncated, info = envs.step(np.asarray(action))
            steps_taken[~done] += 1

            just_done = np.logical_or(terminated, truncated) & ~done
            if just_done.any():
                reported = info.get("final_info")
                for index in np.flatnonzero(just_done):
                    entry = reported[index] if reported is not None else None
                    finals[index] = dict(entry) if entry is not None else {}
                done |= just_done
            if done.all():
                break
        else:
            raise RuntimeError(f"episodes did not finish within {budget} steps")

        for index in range(n_envs):
            final = finals[index]
            best = optimal[index]
            edit = edits[index]
            if final is None or best is None:
                continue
            records.append(
                {
                    "optimal": best,
                    "target": 1 - best,
                    "gap": gaps[index],
                    "reached": bool(final.get("reached_objective")),
                    "reached_feature_id": final.get("reached_feature_id"),
                    "steps": int(steps_taken[index]),
                    "edited_cells": 0 if edit is None else len(edit),
                }
            )
    return records


def outcome_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode ``(switched, reached)`` booleans, aligned across arms.

    Alignment holds because every arm runs the same seeds and drops the same
    episodes — the ones where the two objectives tie — so record ``i`` is the
    same maze in every table. That is what licenses the paired interval.
    """
    reached = np.array([bool(r["reached"]) for r in records])
    switched = np.array([bool(r["reached"]) and r["reached_feature_id"] == r["target"] for r in records])
    return switched, reached


def summarise(records: list[dict], seed: int = 0) -> dict:
    """Switch rate with an interval, reach rate, and how many cells were touched."""
    switched, reached = outcome_arrays(records)
    low, high = metrics.bootstrap_rate(switched, reached, seed=seed)
    return {
        "n": len(records),
        "reached": float(reached.mean()) if records else float("nan"),
        "switched": float(switched.sum() / max(reached.sum(), 1)),
        "switched_ci": [low, high],
        "steps": float(np.mean([r["steps"] for r in records])) if records else float("nan"),
        "cells": float(np.mean([r["edited_cells"] for r in records])) if records else float("nan"),
    }


def switch_by_gap(records: list[dict], edges=(0.0, 0.15, 0.35, 0.75, 10.0)) -> list[tuple[str, int, float]]:
    """Switch rate against how much reward the agent is being asked to give up."""
    reached = [r for r in records if r["reached"]]
    rows = []
    for low, high in zip(edges, edges[1:]):
        group = [r for r in reached if low <= r["gap"] < high]
        if len(group) < 10:
            continue
        rate = sum(r["reached_feature_id"] == r["target"] for r in group) / len(group)
        rows.append((f"{low:.2f}-{high:.2f}", len(group), rate))
    return rows


# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}")

    fixed_values = args.feature_values
    if args.hide_values and fixed_values is None:
        raise SystemExit("--hide-values needs --feature-values: the observation no longer carries them")

    def env_config(seed: int, split: str) -> MazeConfig:
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
            seed=seed,
        )
        if args.levels:
            settings["dataset_split"] = split
        return MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=env_config(0, args.fit_split))
    params = train_state.params
    print(f"checkpoint {args.checkpoint.name}  (update {update})\n")

    # ---- fit the directional probe on the cell state, one probe per layer ----
    fit_rollouts = collect_rollouts(env_config(0, args.fit_split).make(), policy, params, args.fit_episodes, seed=0)
    score_rollouts = collect_rollouts(env_config(1, args.fit_split).make(), policy, params, 256, seed=9999)

    channels = fit_rollouts[0].cell_state.shape[-1] // N_LAYERS
    print(f"cell state {fit_rollouts[0].cell_state.shape} -> {N_LAYERS} layers x {channels} channels")

    def rows(rollouts, layer: int):
        xs, ys = [], []
        for rollout in rollouts:
            labels = plans.observed_directions(rollout)
            keep = labels >= 0
            xs.append(rollout.cell_state[:, :, layer * channels : (layer + 1) * channels][keep])
            ys.append(labels[keep])
        return np.concatenate(xs).astype(np.float64), np.concatenate(ys)

    print(f"\n{'layer':>7}{'accuracy':>11}{'route cells':>13}{'majority':>10}{'min margin':>12}")
    directions: dict[int, np.ndarray] = {}
    fitted: dict[int, tuple] = {}
    for layer in range(N_LAYERS):
        x_fit, y_fit = rows(fit_rollouts, layer)
        x_test, y_test = rows(score_rollouts, layer)
        weights, mean, std = fit_multinomial(x_fit, y_fit, plans.N_CLASSES)
        predicted = apply_multinomial(x_test, weights, mean, std).argmax(1)
        layer_directions, margins = class_directions(weights, std)

        on_route = y_test < plans.NEVER
        majority = max((y_test == k).mean() for k in range(plans.N_CLASSES))
        print(
            f"{layer:>7}{(predicted == y_test).mean():>11.3f}"
            f"{(predicted[on_route] == y_test[on_route]).mean():>13.3f}{majority:>10.3f}{margins.min():>12.3f}"
        )
        if layer in args.layers:
            directions[layer] = layer_directions
            fitted[layer] = (weights, mean, std, x_test)

    counts = plans.class_counts(np.concatenate([plans.observed_directions(r) for r in score_rollouts]))
    print(f"class balance {counts}")
    print(
        "'route cells' is accuracy over the four directions only, where the majority-class floor is\n"
        "25% rather than the NEVER rate. A 1x1 probe on the observation cannot compute a route at all,\n"
        "so anything above chance here is the network's, not the probe's."
    )

    # ---- scale, in units of what is already there ----
    typical = float(
        np.mean(
            [
                np.linalg.norm(r.cell_state[:, :, : channels][geometry.free_cells(r.observation)], axis=-1).mean()
                for r in fit_rollouts[:256]
            ]
        )
    )
    print(f"\ntypical per-cell cell-state norm {typical:.3f}; alpha below is a multiple of it")

    # Does the write say what it claims to? Raising a class's logit is not the
    # same as making the probe predict it, and the scalar work already lost a
    # night to an intervention whose arithmetic was wrong in a way that produced
    # a plausible table rather than a crash.
    print(f"\n{'alpha':>7}" + "".join(f"{'layer %d' % layer:>10}" for layer in sorted(fitted)))
    for alpha in args.alphas:
        if alpha == 0.0:
            continue
        rates = [
            class_write_accuracy(x_test, weights, mean, std, directions[layer], alpha * typical)
            for layer, (weights, mean, std, x_test) in sorted(fitted.items())
        ]
        print(f"{alpha:>7.2f}" + "".join(f"{rate:>10.3f}" for rate in rates))
    print(
        "Fraction of held-out cells where writing a class's direction makes the probe read that class.\n"
        "An alpha whose write accuracy is low is not testing the hypothesis: the network is being handed\n"
        "a vector the probe itself does not call a plan."
    )

    random_table = np.random.default_rng(args.seed).normal(size=(plans.N_CLASSES, channels))
    random_table /= np.linalg.norm(random_table, axis=1, keepdims=True)
    random_directions = {layer: random_table for layer in directions}

    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    envs = env_config(args.seed, args.split).make()

    def run(arm: str, table: dict[int, np.ndarray], alpha: float) -> tuple[dict, list[dict]]:
        records = measure(
            envs, policy, params, get_action, args.batches, args.seed, table, alpha * typical,
            arm, fixed_values, channels,
        )
        return summarise(records, seed=args.seed), records

    def row(label: str, alpha: float, summary: dict) -> str:
        interval = f"[{summary['switched_ci'][0]:.3f},{summary['switched_ci'][1]:.3f}]"
        return (
            f"{label:>10}{alpha:>7.2f}{summary['switched']:>11.1%}{interval:>16}"
            f"{summary['reached']:>10.1%}{summary['steps']:>8.1f}{summary['cells']:>7.1f}{summary['n']:>6}"
        )

    print(
        f"\n{'arm':>10}{'alpha':>7}{'switched':>11}{'95% CI':>16}"
        f"{'reached':>10}{'steps':>8}{'cells':>7}{'n':>6}"
    )
    baseline, baseline_records = run("plan", directions, 0.0)
    print(row("none", 0.0, baseline))
    if baseline["reached"] < 0.9:
        raise RuntimeError(f"unsteered agent reached only {baseline['reached']:.1%}; the rollout loop is not the policy")

    arms = (
        ("plan", directions),
        ("self", directions),
        ("route", directions),
        ("erase", directions),
        ("random", random_directions),
        ("shuffled", directions),
    )
    by_arm: dict[str, list[dict]] = {}
    for arm, table in arms:
        for alpha in args.alphas:
            if alpha == 0.0:
                continue
            summary, records = run(arm, table, alpha)
            print(row(arm, alpha, summary))
            by_arm.setdefault(arm, []).append({"alpha": alpha, **summary, "records": records})
        print()

    print(
        "'switched' is the fraction of episodes ending at the objective an optimal agent would NOT take,\n"
        "among those that ended at an objective at all. The alpha=0 row is the agent's own error rate.\n"
        "'route' writes only the new directions, 'erase' only NEVER along the old route: if erase alone\n"
        "does the work, the edit is a blockade and the directional concept is carrying nothing.\n"
        "Only rows keeping 'reached' near the baseline are interpretable: below that the write is\n"
        "destroying the policy, and 'self' — the same edit pointing where the agent was already going —\n"
        "is what separates the two."
    )

    # Every arm ran the same seeds over the same mazes, so the difference can be
    # taken on a common resample. Two overlapping intervals are not evidence of
    # no difference, and at these rates they overlap constantly.
    baseline_switched, baseline_reached = outcome_arrays(baseline_records)
    print(f"\n{'arm':>10}{'alpha':>7}{'vs baseline':>13}{'95% CI':>18}{'vs self':>10}{'95% CI':>18}")
    for arm in ("plan", "route", "erase", "shuffled", "random"):
        for entry in by_arm.get(arm, []):
            switched, reached = outcome_arrays(entry["records"])
            change, low, high = metrics.bootstrap_rate_difference(
                switched, reached, baseline_switched, baseline_reached, seed=args.seed
            )
            against_self = ""
            twin = next((e for e in by_arm.get("self", []) if e["alpha"] == entry["alpha"]), None)
            if twin is not None and arm != "self":
                other_switched, other_reached = outcome_arrays(twin["records"])
                gap, gap_low, gap_high = metrics.bootstrap_rate_difference(
                    switched, reached, other_switched, other_reached, seed=args.seed
                )
                against_self = f"{gap:>10.1%}" + f"{f'[{gap_low:+.3f},{gap_high:+.3f}]':>18}"
            print(
                f"{arm:>10}{entry['alpha']:>7.2f}{change:>13.1%}"
                + f"{f'[{low:+.3f},{high:+.3f}]':>18}"
                + against_self
            )
    print(
        "An interval excluding zero is a real shift. 'vs self' is the strongest form of the comparison:\n"
        "identical machinery, identical cells, identical norm, pointed the other way."
    )

    if args.per_layer and len(directions) > 1:
        # The paper reports intervention success per layer, and it is the one
        # place a DRC's depth is legible: if a single layer carries the plan the
        # policy reads, writing to it alone should be enough.
        print(f"\n{'layer':>10}{'alpha':>7}{'switched':>11}{'reached':>10}{'steps':>8}{'n':>6}")
        for layer in sorted(directions):
            for alpha in args.alphas:
                if alpha == 0.0:
                    continue
                summary, _ = run("plan", {layer: directions[layer]}, alpha)
                print(
                    f"{layer:>10}{alpha:>7.2f}{summary['switched']:>11.1%}"
                    f"{summary['reached']:>10.1%}{summary['steps']:>8.1f}{summary['n']:>6}"
                )
            print()

    for arm in ("plan", "self", "random", "shuffled"):
        for entry in by_arm.get(arm, []):
            if entry["reached"] < 0.9 * baseline["reached"]:
                continue
            table = switch_by_gap(entry["records"])
            if not table:
                continue
            print(f"\n{arm} at alpha {entry['alpha']:.2f}, switch rate by utility given up:")
            for label, count, rate in table:
                print(f"  {label:>12}{count:>7}{rate:>9.1%}")
    print(
        "\nThe gap is what the agent is being asked to give up, in reward. A plan that is followed\n"
        "when it is cheap and refused when it is expensive is being weighed against something; one\n"
        "that is followed regardless is overwriting the decision rather than entering it."
    )


    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "commit": commit,
                    "checkpoint": str(args.checkpoint),
                    "update": update,
                    "episodes": args.batches * args.num_envs,
                    "typical_cell_state_norm": typical,
                    "baseline": {k: v for k, v in baseline.items() if k != "records"},
                    "arms": {
                        arm: [{k: v for k, v in entry.items() if k != "records"} for entry in entries]
                        for arm, entries in by_arm.items()
                    },
                    "by_gap": {
                        arm: {
                            f"{entry['alpha']:.2f}": [
                                {"band": label, "n": count, "switched": rate}
                                for label, count, rate in switch_by_gap(entry["records"])
                            ]
                            for entry in entries
                        }
                        for arm, entries in by_arm.items()
                    },
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
