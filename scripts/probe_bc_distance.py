"""Probe the route model's residual stream for its own distance estimates.

    uv run python scripts/probe_bc_distance.py /workspace/data/offline/runs/bcnv11.s1 \
        --choices figures/data/h1/bcnv11.s1.npz > results/gap-probe-bcnv11.s1.txt

The experiment UtilityRule.md part 3 ends on: behaviour cannot tell a wobbling
threshold from a misread gap, so read the gap out of the activations and ask
which. Ridge probes are trained to predict the true ``d_rich``, ``d_poor`` and
their difference from the prefix residuals - on levels disjoint from the
evaluation set, and never on choice - then scored three ways on the held-out
levels:

1. How well each (site, depth) reads the true quantities. This calibrates the
   probe; it is not the result.
2. Within a fixed ``(d_rich, d_poor)`` cell, does the probe's *residual*
   predict choice? The target is constant inside a cell, so anything the
   residual knows about choice is the model's own misread leaking through.
   The yardstick is the 21 hand-built level features at r2 ~ 0.01.
3. Split-half reliability: two probes on disjoint training halves. The
   correlation of their eval residuals bounds how much of the residual is a
   stable readout rather than probe noise, and disattenuates statistic 2 -
   an estimate of what a noiseless probe would see.

Values are hidden throughout, matching how ``bcnv11`` was trained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

import goalmisgen.provenance as provenance
from goalmisgen.offline.demos import DemoSet, shared_levels
from goalmisgen.offline.gap_probe import (
    SITES,
    cell_members,
    centre_within_cells,
    collect_site_features,
    fit_gap_probe,
    flatten_depths,
    gap_targets,
    within_cell_choice,
)
from goalmisgen.offline.train import initial_params, list_checkpoints, load_checkpoint

DEFAULT_PROBE_DEMOS = "/workspace/data/offline/demos/train.rho100"
DEFAULT_EVAL_DEMOS = "/workspace/data/offline/demos/test.rho100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="A route-model run directory; its last checkpoint is used.")
    parser.add_argument("--probe-demos", type=str, default=DEFAULT_PROBE_DEMOS, help="Levels the probes train on.")
    parser.add_argument("--eval-demos", type=str, default=DEFAULT_EVAL_DEMOS, help="Levels everything is scored on.")
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-eval", type=int, default=50_000)
    parser.add_argument(
        "--choices",
        type=Path,
        default=None,
        help="A decode_h1-style .npz aligned with the first n-eval levels of eval-demos; decoded here if absent.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-n", type=int, default=40, help="Smallest (d_rich, d_poor) cell the within-cell statistics use.")
    return parser.parse_args()


def load_choices(args, model, params, demos: DemoSet, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per level: did the model take the richer objective, and did it reach anything."""
    if args.choices is not None:
        data = np.load(args.choices)
        if len(data["reached_fid"]) < len(indices):
            raise SystemExit(f"{args.choices} holds {len(data['reached_fid'])} levels, fewer than n-eval")
        reached_fid = data["reached_fid"][: len(indices)]
        colour_of_rich = data["colour_of_rich"][: len(indices)]
        return reached_fid == colour_of_rich, data["reached"][: len(indices)].astype(bool)

    from goalmisgen.offline.decode import evaluate
    from goalmisgen.offline.fast_decode import greedy_decode_cached

    _, _, outcomes = evaluate(model, params, demos, indices, decoder=greedy_decode_cached, indifference=False)
    values = np.asarray(demos.values)[indices]
    feature_ids = np.asarray(demos.feature_ids)[indices]
    colour_of_rich = feature_ids[np.arange(len(indices)), np.argmax(values, axis=1)]
    reached_fid = np.array([-1 if o.get("reached_feature_id") is None else int(o["reached_feature_id"]) for o in outcomes])
    reached = np.array([bool(o.get("reached_objective")) for o in outcomes])
    return reached_fid == colour_of_rich, reached


def main() -> None:
    args = parse_args()
    print(provenance.header())
    print()

    probe_demos = DemoSet.load(args.probe_demos, hide_values=True)
    eval_demos = DemoSet.load(args.eval_demos, hide_values=True)
    overlap = shared_levels(probe_demos, eval_demos)
    if overlap:
        raise SystemExit(f"probe and eval demos share {overlap} levels; the probe would be scored on its training data")

    model, params = load_checkpoint(list_checkpoints(args.run)[-1][1])
    untrained = initial_params(model, jax.random.PRNGKey(1))

    train_idx = np.arange(min(args.n_train, len(probe_demos)))
    eval_idx = np.arange(min(args.n_eval, len(eval_demos)))
    train_targets = gap_targets(probe_demos, train_idx)
    eval_targets = gap_targets(eval_demos, eval_idx)
    choice, reached = load_choices(args, model, params, eval_demos, eval_idx)

    train_ok = train_targets.valid
    eval_ok = eval_targets.valid
    scored = eval_ok & reached  # choice is only defined where something was reached
    print(f"probe-train  {train_ok.sum():,} of {len(train_idx):,} levels of {args.probe_demos}")
    print(f"eval         {eval_ok.sum():,} valid, {scored.sum():,} scored, of {args.eval_demos}")
    print(f"take rate    {choice[scored].mean():.3f}")
    print()

    targets = {
        "gap": (train_targets.gap, eval_targets.gap),
        "d_rich": (train_targets.d_rich, eval_targets.d_rich),
        "d_poor": (train_targets.d_poor, eval_targets.d_poor),
    }
    depths = model.config.n_layers + 1
    layers: list[int | None] = [*range(depths), None]

    def probe_arm(name: str, train_x: dict[str, np.ndarray], eval_x: dict[str, np.ndarray], full_grid: bool) -> tuple[dict, dict]:
        """Fit every (target, site, layer); return the gap predictions and r2s for stage 2."""
        gap_predictions, gap_r2 = {}, {}
        for target, (y_train, y_eval) in targets.items():
            if not full_grid and target != "gap":
                continue
            print(f"=== {name}: probing {target} ===")
            print(f"  {'site':<12} " + " ".join(f"{f'layer {l}' if l is not None else 'all':>24}" for l in layers))
            for site in SITES:
                row = []
                for layer in layers:
                    result = fit_gap_probe(
                        flatten_depths(train_x[site], layer)[train_ok],
                        y_train[train_ok].astype(np.float64),
                        flatten_depths(eval_x[site], layer)[eval_ok],
                        y_eval[eval_ok].astype(np.float64),
                    )
                    row.append(f"r2 {result.r2:+.3f} sd {result.residual_std:5.2f}")
                    if target == "gap":
                        predictions = np.full(len(eval_idx), np.nan)
                        predictions[eval_ok] = result.predictions
                        gap_predictions[(site, layer)] = predictions
                        gap_r2[(site, layer)] = result.r2
                print(f"  {site:<12} " + " ".join(f"{cell:>24}" for cell in row))
            print()
        return gap_predictions, gap_r2

    def within_cell_table(name: str, gap_predictions: dict) -> None:
        print(f"=== {name}: does the probed gap predict choice within a (d_rich, d_poor) cell? ===")
        print("  Target is constant inside a cell, so this is the probe residual against the model's choice.")
        print("  Yardstick: 21 hand-built level features reach r2 ~ 0.01 (utility-rule-bcnv11.txt part 4).")
        for (site, layer), predictions in gap_predictions.items():
            keep = scored & np.isfinite(predictions)
            result = within_cell_choice(
                eval_targets.d_rich[keep], eval_targets.d_poor[keep], predictions[keep], choice[keep], min_n=args.min_n
            )
            print(f"  {site:<12} {'all' if layer is None else f'layer {layer}':<8} {result.summary}")
        print()

    train_x = collect_site_features(model, params, probe_demos, train_idx, batch_size=args.batch_size)
    eval_x = collect_site_features(model, params, eval_demos, eval_idx, batch_size=args.batch_size)
    gap_predictions, gap_r2 = probe_arm("trained", train_x, eval_x, full_grid=True)
    within_cell_table("trained", gap_predictions)
    # The reliability configuration is picked by probe calibration alone, never by
    # its choice statistic - selecting on the outcome would bias the estimate up.
    best_key = max(gap_r2, key=gap_r2.get) if gap_r2 else None

    # --- controls -----------------------------------------------------------
    train_u = collect_site_features(model, untrained, probe_demos, train_idx, batch_size=args.batch_size)
    eval_u = collect_site_features(model, untrained, eval_demos, eval_idx, batch_size=args.batch_size)
    untrained_predictions, _ = probe_arm("untrained network", train_u, eval_u, full_grid=False)
    within_cell_table("untrained network (same routes)", untrained_predictions)

    print("=== observation baseline: a linear readout of the raw maze ===")
    y_train, y_eval = targets["gap"]
    result = fit_gap_probe(
        probe_demos.observations(train_idx).reshape(len(train_idx), -1)[train_ok],
        y_train[train_ok].astype(np.float64),
        eval_demos.observations(eval_idx).reshape(len(eval_idx), -1)[eval_ok],
        y_eval[eval_ok].astype(np.float64),
    )
    predictions = np.full(len(eval_idx), np.nan)
    predictions[eval_ok] = result.predictions
    keep = scored & np.isfinite(predictions)
    cell = within_cell_choice(eval_targets.d_rich[keep], eval_targets.d_poor[keep], predictions[keep], choice[keep], min_n=args.min_n)
    print(f"  gap: {result.summary}")
    print(f"  within-cell: {cell.summary}")
    print("  A linear map cannot run BFS; this bounds what maze appearance alone buys.")
    print()

    # --- split-half reliability at the best configuration -------------------
    if best_key is not None:
        site, layer = best_key
        print(f"=== split-half reliability at {site} / {'all' if layer is None else f'layer {layer}'} ===")
        order = np.random.default_rng(0).permutation(np.flatnonzero(train_ok))
        halves = [np.sort(order[::2]), np.sort(order[1::2])]
        half_predictions, choice_rs = [], []
        for half in halves:
            result = fit_gap_probe(
                flatten_depths(train_x[site], layer)[half],
                y_train[half].astype(np.float64),
                flatten_depths(eval_x[site], layer)[eval_ok],
                y_eval[eval_ok].astype(np.float64),
            )
            predictions = np.full(len(eval_idx), np.nan)
            predictions[eval_ok] = result.predictions
            keep = scored & np.isfinite(predictions)
            cell = within_cell_choice(eval_targets.d_rich[keep], eval_targets.d_poor[keep], predictions[keep], choice[keep], min_n=args.min_n)
            half_predictions.append(predictions)
            choice_rs.append(np.sqrt(cell.r2) if np.isfinite(cell.r2) else np.nan)
            print(f"  half probe: gap {result.summary}; within-cell {cell.summary}")

        keep = scored & np.isfinite(half_predictions[0]) & np.isfinite(half_predictions[1])
        cells = cell_members(eval_targets.d_rich[keep], eval_targets.d_poor[keep], min_n=args.min_n)
        a, _ = centre_within_cells(half_predictions[0][keep], cells)
        b, _ = centre_within_cells(half_predictions[1][keep], cells)
        reliability = float(np.corrcoef(a, b)[0, 1]) if len(a) else float("nan")
        mean_r = float(np.nanmean(choice_rs))
        print(f"  within-cell corr(residual A, residual B)  {reliability:+.3f}   (probe signal against probe noise)")
        if reliability > 0:
            print(f"  disattenuated choice correlation          {mean_r / np.sqrt(reliability):+.3f}   (what a noiseless probe would see)")
        print("  Squared, the disattenuated value is the share of within-cell choice variance the model's")
        print("  own gap estimate accounts for. Near the cross-seed xi ceiling, the misread is all of xi.")


if __name__ == "__main__":
    main()
