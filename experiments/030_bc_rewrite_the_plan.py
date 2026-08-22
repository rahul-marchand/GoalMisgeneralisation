"""Rewrite the route model's plan in its residual stream, and see if it walks the new one.

    uv run python experiments/030_bc_rewrite_the_plan.py RUN_DIR \\
        --demos /workspace/data/offline/demos/test.rho050 [--depths 1 2 3 4] [--patch]

The offline twin of ``012_rewrite_the_plan.py``, and the half of Experiment 1
the BC stream has not had. ``025`` ported the reading half: the route the model
will walk is decodable from the per-cell residual before any action token, at
0.74 AUC against 0.60 for an untrained network of the same shape - the same
sign as the DRC's 0.967 against 0.583, and much weaker. This asks the causal
half. Fit the directional probe on the model's own routes, build the route to
the objective it is *not* taking, write it into the residual at one depth, and
decode.

Three things differ from the DRC, and the third decides how a null must be read.

**The write persists for free.** The prefix is recomputed identically at every
decoded token and never sees an action token, so one edit inside the forward
pass is present for the whole route. No re-application, and no question about
whether the edit arrived before the action was chosen.

**The failure modes are a decoder's.** A damaged route model walks into walls or
never emits EOS, neither of which an agent stepping an environment can do.
``legal`` and ``eos`` are reported beside ``reached``, and an alpha where they
fall is an alpha whose switch rate is measuring damage.

**The write has a depth budget.** A maze-token edit reaches the logits only
through attention in later blocks, so depth ``n_layers`` is a *guaranteed* null
- the head reads from SEP onward - and depth ``n_layers - 1`` has one block to
be consumed in, against the DRC's three ticks in every remaining step. The probe
reads best at the top and the write can only act from below, so the band where
both hold is narrow, and that is what the depth sweep is for. Before any
behaviour is read, two checks say whether the arithmetic is testing the
hypothesis at all: the probe reads back the class written at the site
(``write-back``), and the probe at the top still reads it after an edit made
lower down (``propagation``).

Registered before running:

* the plan direction switches the objective and the norm-matched controls do
  not -> the DRC's planning result replicates under imitation
* the plan direction does no better than ``self`` or ``random`` while write-back
  and propagation are high -> the route is decodable and not used, which is a
  finding about imitation rather than about the method
* write-back or propagation is low -> the run measured the depth budget, and
  ``--patch`` says what the site could have done

Arms, all at identical norms on identical cells, as in ``012``:

``plan``      the route to the other objective, ``NEVER`` along the one replaced
``route``     only the new route, with nothing erased
``erase``     only the ``NEVER`` half: "not that way", with no plan at all
``self``      the same machinery pointed where the model was already going
``shuffled``  a real plan, for a different level's maze
``random``    fixed random unit vectors in place of the probe's class directions

``--patch`` is a ceiling rather than an arm. It replaces the per-cell residual
with the one the same maze produces when the two objectives' values are swapped,
which is the largest coherent edit that depth can carry, and optionally only at
the two objective cells. A ``plan`` null under a working patch is about the
fitted direction; a null under both is about the site.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen import provenance
from goalmisgen.analysis import metrics
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.probes import class_directions
from goalmisgen.offline import rewrite
from goalmisgen.offline.decode import greedy_decode, replay_all, summarise_routes
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.probe import capture, cell_residuals
from goalmisgen.offline.train import list_checkpoints, load_checkpoint, load_run_config

ARMS = ("plan", "route", "erase", "self", "shuffled", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run directory of the route model.")
    parser.add_argument("--demos", type=Path, required=True, help="Held-out demonstrations to intervene on.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step; default the last.")
    parser.add_argument("--fit-episodes", type=int, default=512, help="Levels the directional probe is fitted on.")
    parser.add_argument("--episodes", type=int, default=1024, help="Levels each arm is measured on.")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="*",
        default=None,
        help="Where to write; 0 is the embedding, k is after block k. Default every depth, including "
        "n_layers, whose null is guaranteed and which is run as the control it is.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
        help="Norm of the written vector, as a multiple of the typical per-cell residual norm at that depth. "
        "The band is far lower than the DRC's: at a fifth of the residual's own norm the probe already reads "
        "the written class back perfectly, and at the residual's full norm the decoder is walking into walls.",
    )
    parser.add_argument("--arms", type=str, nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--patch", action="store_true", help="Also run the counterfactual patch ceiling.")
    parser.add_argument(
        "--counterfactual",
        choices=("auto", "values", "features"),
        default="auto",
        help="What the patch changes about the level it is taken from. 'auto' swaps the values, or the "
        "colours when the model was trained without a value channel and cannot see a value change.",
    )
    parser.add_argument(
        "--patch-cells",
        choices=("all", "objectives"),
        default="all",
        help="Patch every maze cell, or only the two the objectives sit on.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


# ----------------------------------------------------------------------
# Outcomes
# ----------------------------------------------------------------------


def records_from(outcomes: list[dict], edits: list[dict | None], targets: list[int | None]) -> list[dict]:
    """One row per level, aligned across arms: every arm decodes the same levels."""
    rows = []
    for outcome, edit, target in zip(outcomes, edits, targets):
        if target is None:
            continue
        rows.append(
            {
                "target": target,
                "gap": float(abs(outcome["utility_margin"])),
                "reached": bool(outcome["reached_objective"]),
                "reached_feature_id": outcome.get("reached_feature_id"),
                "legal": bool(outcome["illegal_moves"] == 0),
                "emitted_eos": bool(outcome["emitted_eos"]),
                "steps": int(outcome["episode_steps"]),
                "edited_cells": 0 if edit is None else len(edit),
                # Kept whole so the exchange rate can be refitted per arm: the
                # switch rate is a percentage, the exchange rate is in steps.
                "info": outcome,
            }
        )
    return rows


def switched_and_reached(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    switched = np.array([bool(r["reached"]) and r["reached_feature_id"] == r["target"] for r in records])
    return switched, np.array([bool(r["reached"]) for r in records])


def exchange_rate(records: list[dict], resamples: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """The distance gap at which the richer objective stops being preferred, in steps.

    A switch rate is a percentage of a population the intervention reshapes;
    this is a property of the decision, and comparable with the expert's 10.0
    and with the DRC numbers in ``results/``.
    """
    gaps, took_richer, _ = value_distance_decisions([r["info"] for r in records])
    if not len(gaps):
        return float("nan"), float("nan"), float("nan")
    point = indifference_point(gaps, took_richer)

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(gaps), len(gaps))
        resampled = indifference_point(gaps[chosen], took_richer[chosen])
        if np.isfinite(resampled):
            values.append(resampled)
    if not values:
        return point, float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def summarise(records: list[dict], seed: int = 0) -> dict:
    """Switch rate and exchange rate with intervals, plus what the edit cost."""
    switched, reached = switched_and_reached(records)
    low, high = metrics.bootstrap_rate(switched, reached, seed=seed)
    point, point_low, point_high = exchange_rate(records, seed=seed)
    return {
        "n": len(records),
        "reached": float(reached.mean()) if len(records) else float("nan"),
        "legal": float(np.mean([r["legal"] for r in records])) if records else float("nan"),
        "eos": float(np.mean([r["emitted_eos"] for r in records])) if records else float("nan"),
        "switched": float(switched.sum() / max(reached.sum(), 1)),
        "switched_ci": [low, high],
        "indifference": point,
        "indifference_ci": [point_low, point_high],
        "steps": float(np.mean([r["steps"] for r in records])) if records else float("nan"),
        "cells": float(np.mean([r["edited_cells"] for r in records])) if records else float("nan"),
    }


def paired_difference(records: list[dict], reference: list[dict], resamples: int = 2000, seed: int = 0):
    """Switch rate minus the reference's, resampled on the levels they share.

    Every arm ran the same levels and dropped the same ties, so record ``i`` is
    the same maze in both tables and the between-level variance cancels. Two
    overlapping marginal intervals are not evidence of no difference, and at
    these rates they overlap constantly.
    """
    switched, reached = switched_and_reached(records)
    base_switched, base_reached = switched_and_reached(reference)
    difference, low, high = metrics.bootstrap_rate_difference(
        switched, reached, base_switched, base_reached, resamples=resamples, seed=seed
    )
    return float(difference), float(low), float(high)


def switch_by_gap(records: list[dict], edges=(0.0, 0.15, 0.35, 0.75, 10.0)) -> list[tuple[str, int, float]]:
    """Switch rate against how much reward the write asks the model to give up."""
    reached = [r for r in records if r["reached"]]
    rows = []
    for low, high in zip(edges, edges[1:]):
        group = [r for r in reached if low <= r["gap"] < high]
        if len(group) < 10:
            continue
        rate = sum(r["reached_feature_id"] == r["target"] for r in group) / len(group)
        rows.append((f"{low:.2f}-{high:.2f}", len(group), rate))
    return rows


def strongest_usable(rows: list[dict], floor: float = 0.95) -> dict | None:
    """The largest alpha that still leaves the decoder competent.

    Past the point where routes stop being legal or stop terminating, a switch
    rate describes a broken decoder rather than a moved decision, which is why
    the DRC's figure dots its line there.
    """
    usable = [row for row in rows if row["reached"] >= floor and row["legal"] >= floor]
    return max(usable, key=lambda row: row["alpha"]) if usable else None


# ----------------------------------------------------------------------
# The arms
# ----------------------------------------------------------------------


def build_edits(arm: str, observations, outcomes, targets, seed: int) -> list[dict | None]:
    """The cells and classes this arm writes, one entry per measured level."""
    built: list[dict | None] = []
    for observation, outcome, target in zip(observations, outcomes, targets):
        if target is None:
            built.append(None)
            continue
        optimal = int(outcome["optimal_feature_id"])
        # "self" points the same machinery where the model was already going.
        pointed_at, replaced = (optimal, target) if arm == "self" else (target, optimal)
        built.append(
            rewrite.plan_edit(
                observation,
                pointed_at,
                replaced,
                write_route=arm != "erase",
                erase_old=arm != "route",
            )
        )
    if arm == "shuffled":
        built = [built[index] for index in rewrite.derange(len(built), seed)]
    return built


def counterfactual_demos(demos: DemoSet, args, hide_values: bool) -> tuple[DemoSet, str]:
    """The level the patch is taken from, and what was changed to make it."""
    kind = args.counterfactual
    if kind == "auto":
        kind = "features" if hide_values else "values"
    if kind == "values":
        return rewrite.swapped_values(demos), "values"
    return rewrite.swapped_features(demos), "colours"


def run_patch(model, params, demos, indices, observations, targets, depth, args, hide_values: bool) -> dict:
    """The ceiling: replace the residual with the one the counterfactual maze makes."""
    source, kind = counterfactual_demos(demos, args, hide_values)
    counterfactual = source.observations(indices)
    before = cell_residuals(model, params, observations)[depth]
    after = cell_residuals(model, params, counterfactual)[depth]
    cells = None
    if args.patch_cells == "objectives":
        cells = [rewrite.objective_cells(observation) for observation in observations]
    grid = rewrite.patch_edit(before, after, cells)

    patched = greedy_decode(model, params, observations, edit=grid, edit_depth=depth)
    entry = summarise(
        records_from(replay_all(demos, indices, patched), [None] * len(observations), targets), seed=args.seed
    )
    interval = f"[{entry['switched_ci'][0]:.3f},{entry['switched_ci'][1]:.3f}]"
    print(
        f"\npatch ({kind} swapped, {args.patch_cells} cells): switched {entry['switched']:.1%} {interval}  "
        f"reached {entry['reached']:.1%}  legal {entry['legal']:.1%}  indifference {entry['indifference']:.2f}"
    )
    print(
        "The largest coherent edit this depth can carry. A plan null under a working patch is about the\n"
        "fitted direction; a null under both is about the site."
    )
    return entry


# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    print(provenance.header())
    print()

    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.step is None else next(c for c in checkpoints if c[0] == args.step)
    model, params = load_checkpoint(directory)
    cfg = model.config
    hide_values = bool(load_run_config(args.run)["demos"].get("hide_values", False))
    demos = DemoSet.load(args.demos, hide_values=hide_values)
    depths = list(args.depths) if args.depths else list(range(cfg.n_layers + 1))
    rng = np.random.default_rng(args.seed)

    fit_indices = np.arange(args.fit_episodes)
    measure_indices = np.arange(args.fit_episodes, args.fit_episodes + args.episodes)
    observations = demos.observations(measure_indices)
    print(
        f"{args.run.name} @ step {step:,}; {args.demos.name} (rho={demos.rho}); "
        f"{args.fit_episodes} fitted / {args.episodes} measured; {cfg.n_layers} blocks, d_model {cfg.d_model}; "
        f"values {'hidden' if hide_values else 'shown'}"
    )

    # ---- the baseline the switch rate is a departure from --------------------
    decoded = greedy_decode(model, params, observations)
    outcomes = replay_all(demos, measure_indices, decoded)
    print(f"\nunedited: {summarise_routes(demos, measure_indices, decoded, outcomes)}")
    if summarise_routes(demos, measure_indices, decoded, outcomes).behaviour.reached_objective < 0.9:
        raise RuntimeError("the unedited model reaches under 90%; the decode path is not the model")

    # The write asks for the objective the model would not take. Levels where
    # the two are worth the same are dropped rather than tie-broken: a switch
    # there would be a success under either label.
    targets: list[int | None] = [
        None if outcome["is_ambiguous"] else int(1 - outcome["optimal_feature_id"]) for outcome in outcomes
    ]
    print(f"{sum(t is None for t in targets)} of {len(targets)} levels dropped as ties")

    edits = {arm: build_edits(arm, observations, outcomes, targets, args.seed) for arm in args.arms}
    print("cells written  " + "  ".join(f"{arm} {np.mean([0 if e is None else len(e) for e in edits[arm]]):.1f}" for arm in args.arms))

    # The probe the propagation check is read with: the top of the stack, where
    # 025 finds the route most decodable and where nothing can be written.
    read_depth = cfg.n_layers
    top_rollouts = capture(model, params, demos, fit_indices, layer=read_depth)
    top_probe = rewrite.fit_plan_probe(top_rollouts)

    results: dict = {
        "run": args.run.name,
        "step": step,
        "demos": args.demos.name,
        "episodes": int(args.episodes),
        "depths": {},
    }

    for depth in depths:
        print(f"\n{'=' * 78}")
        print(f"depth {depth}" + ("  (embedding)" if depth == 0 else f"  (after block {depth})"))
        if depth == cfg.n_layers:
            print(
                "The head reads from SEP onward and no attention layer follows, so an edit here cannot\n"
                "change a logit. Every arm must read exactly the unedited row; anything else is a bug."
            )

        fit_rollouts = top_rollouts if depth == read_depth else capture(model, params, demos, fit_indices, layer=depth)
        weights, mean, std = rewrite.fit_plan_probe(fit_rollouts)
        overall, on_route, majority = rewrite.probe_accuracy(fit_rollouts, weights, mean, std)
        try:
            directions, margins = class_directions(weights, std)
        except ValueError as unwritable:
            # A class whose weights lie inside the cone of the others cannot be
            # written by any additive edit. That is a fact about this depth's
            # readout, worth reporting and moving past rather than ending a run
            # that has other depths to measure.
            print(f"no write is possible at this depth: {unwritable}")
            results["depths"][str(depth)] = {"unwritable": str(unwritable)}
            continue
        typical = rewrite.typical_cell_norm(fit_rollouts)
        print(
            f"probe: accuracy {overall:.3f}  route cells {on_route:.3f}  majority {majority:.3f}  "
            f"min margin {margins.min():.3f}"
        )
        print(f"typical per-cell residual norm {typical:.3f}; alpha below is a multiple of it")

        measured = capture(model, params, demos, measure_indices, layer=depth, decoded=decoded)
        top_before = cell_residuals(model, params, observations)[read_depth]
        random_directions = rng.normal(size=directions.shape)
        random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)

        # Does the write say what it claims to? Raising a class's logit is not
        # the same as making the probe read it, and a wrong raster order writes
        # a coherent plan onto the wrong maze and reports a clean null.
        print(f"\n{'alpha':>7}{'write-back':>12}{'propagation':>13}")
        write_back: dict[float, tuple[float, float]] = {}
        for alpha in args.alphas:
            if alpha == 0.0:
                continue
            back = rewrite.written_classes(measured, edits["plan"], weights, mean, std, directions, alpha * typical)
            grid = rewrite.delta_grid(edits["plan"], directions, cfg.size, alpha * typical)
            top_after = cell_residuals(model, params, observations, edit=grid, edit_depth=depth)[read_depth]
            _, carried = rewrite.propagation(top_before, top_after, edits["plan"], *top_probe)
            write_back[alpha] = (back, carried)
            print(f"{alpha:>7.2f}{back:>12.3f}{carried:>13.3f}")
        _, unedited_read = rewrite.propagation(top_before, top_before, edits["plan"], *top_probe)
        print(
            f"write-back is the probe at this depth reading the class written; propagation is the probe at\n"
            f"the top reading it after the edit was made here, against {unedited_read:.3f} with no edit at all.\n"
            "An alpha low on either is not testing the hypothesis: the network is handed a vector the\n"
            "readout does not call the plan, or one that does not survive to where it could be used."
        )

        depth_rows: dict[str, list[dict]] = {}
        baseline_records = records_from(outcomes, edits[args.arms[0]], targets)
        baseline = summarise(baseline_records, seed=args.seed)
        depth_rows["none"] = [{"alpha": 0.0, **baseline}]
        header = (
            f"\n{'arm':>10}{'alpha':>7}{'switched':>10}{'95% CI':>16}"
            f"{'reached':>9}{'legal':>7}{'eos':>7}{'indiff':>8}{'steps':>7}"
        )
        print(header)

        def row(label: str, alpha: float, entry: dict) -> str:
            interval = f"[{entry['switched_ci'][0]:.3f},{entry['switched_ci'][1]:.3f}]"
            return (
                f"{label:>10}{alpha:>7.2f}{entry['switched']:>10.1%}{interval:>16}"
                f"{entry['reached']:>9.1%}{entry['legal']:>7.1%}{entry['eos']:>7.1%}"
                f"{entry['indifference']:>8.2f}{entry['steps']:>7.1f}"
            )

        print(row("none", 0.0, baseline))
        for arm in args.arms:
            table = random_directions if arm == "random" else directions
            for alpha in args.alphas:
                if alpha == 0.0:
                    continue
                grid = rewrite.delta_grid(edits[arm], table, cfg.size, alpha * typical)
                armed = greedy_decode(model, params, observations, edit=grid, edit_depth=depth)
                records = records_from(replay_all(demos, measure_indices, armed), edits[arm], targets)
                entry = summarise(records, seed=args.seed)
                depth_rows.setdefault(arm, []).append({"alpha": alpha, **entry, "records": records})
                print(row(arm, alpha, entry))
            print()

        print(
            "'switched' is the fraction of routes ending at the objective an optimal agent would NOT take,\n"
            "among those ending at an objective at all. The alpha=0 row is the model's own error rate."
        )

        print(f"\n{'arm':>10}{'alpha':>7}{'vs unedited':>13}{'95% CI':>18}{'vs self':>10}{'95% CI':>18}")
        for arm in args.arms:
            for entry in depth_rows.get(arm, []):
                change, low, high = paired_difference(entry["records"], baseline_records, seed=args.seed)
                against_self = ""
                twin = next((e for e in depth_rows.get("self", []) if e["alpha"] == entry["alpha"]), None)
                if twin is not None and arm != "self":
                    other, self_low, self_high = paired_difference(entry["records"], twin["records"], seed=args.seed)
                    against_self = f"{other:>10.1%}{f'[{self_low:+.3f},{self_high:+.3f}]':>18}"
                print(f"{arm:>10}{entry['alpha']:>7.2f}{change:>13.1%}{f'[{low:+.3f},{high:+.3f}]':>18}{against_self}")

        strongest = strongest_usable(depth_rows.get("plan", []))
        if strongest is not None:
            print(f"\nplan at alpha {strongest['alpha']:.2f}, switch rate by utility given up:")
            for label, count, rate in switch_by_gap(strongest["records"]):
                print(f"  {label:>12}{count:>7}{rate:>9.1%}")

        results["depths"][str(depth)] = {
            "probe": {"accuracy": overall, "route_cells": on_route, "majority": majority},
            "typical_norm": typical,
            "write_back": {f"{alpha:.2f}": values for alpha, values in write_back.items()},
            "unedited_read": unedited_read,
            "arms": {
                arm: [{k: v for k, v in entry.items() if k != "records"} for entry in rows]
                for arm, rows in depth_rows.items()
            },
        }
        if args.patch:
            results["depths"][str(depth)]["patch"] = run_patch(
                model, params, demos, measure_indices, observations, targets, depth, args, hide_values
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
