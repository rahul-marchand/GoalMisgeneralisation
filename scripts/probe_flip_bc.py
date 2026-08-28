"""Do the activations hold the model's own gap estimate, or the true one?

    uv run python scripts/probe_flip_bc.py /workspace/data/offline/runs/bcnv11.s1 \
        --flip figures/data/h1/flip/bcnv11.s1.npz > results/probe-flip-bcnv11.s1.txt

The flip points (``scripts/flip_points_bc.py``) give each level's effective gap
as the model reads it - ``gap * (1 + eps)`` - measured behaviourally. That
turns the probe question from predicting a 1-bit choice into predicting a
continuous target, and lets the truth and the model's estimate compete:

1. **Residual tracking.** A probe trained on the *true* gap (disjoint levels)
   is applied to the evaluation set. Within a fixed ``(d_rich, d_poor)`` cell
   its prediction varies only by its residual; the statistic is the
   correlation of that residual with the flip residual ``flip - gap``. This is
   the continuous version of the within-cell choice AUC.
2. **Target swap.** On the bracketed levels, half train / half test: one ridge
   probe trained on the true gap, one on the flip point, both scored on both
   targets. If the activations encode the model's estimate rather than the
   truth, the flip-trained probe wins on flip and the transfer is asymmetric.

Alignment is asserted, not assumed: the flip file stores the true gaps it was
computed against, and they must match the demonstration set's, level for level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from goalmisgen.offline.demos import DemoSet, shared_levels
from goalmisgen.offline.gap_probe import (
    SITES,
    cell_members,
    centre_within_cells,
    collect_site_features,
    fit_gap_probe,
    flatten_depths,
    gap_targets,
)
from goalmisgen.offline.train import initial_params, list_checkpoints, load_checkpoint
from goalmisgen.provenance import header

DEFAULT_PROBE_DEMOS = "/workspace/data/offline/demos/train.rho100"
DEFAULT_EVAL_DEMOS = "/workspace/data/offline/demos/test.rho100"
BAND = (4, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--flip", type=Path, required=True, help="Flip targets from scripts/flip_points_bc.py.")
    parser.add_argument("--probe-demos", type=str, default=DEFAULT_PROBE_DEMOS)
    parser.add_argument("--eval-demos", type=str, default=DEFAULT_EVAL_DEMOS)
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-n", type=int, default=20, help="Smallest cell used for within-cell centring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    flip_data = np.load(args.flip)
    flip, flip_gap = flip_data["flip"], flip_data["gap"]
    n_eval = len(flip)

    probe_demos = DemoSet.load(args.probe_demos, hide_values=True)
    eval_demos = DemoSet.load(args.eval_demos, hide_values=True)
    if shared_levels(probe_demos, eval_demos):
        raise SystemExit("probe and eval demos share levels")

    train_idx = np.arange(min(args.n_train, len(probe_demos)))
    eval_idx = np.arange(n_eval)
    train_targets = gap_targets(probe_demos, train_idx)
    eval_targets = gap_targets(eval_demos, eval_idx)
    if not np.array_equal(flip_gap, eval_targets.gap):
        raise SystemExit("flip file's gaps disagree with the demonstration set; levels are misaligned")

    band = np.isfinite(flip) & (eval_targets.gap >= BAND[0]) & (eval_targets.gap <= BAND[1]) & eval_targets.valid
    flip_residual = flip - eval_targets.gap
    print(f"eval: {n_eval:,} levels, {band.sum():,} bracketed in gap {BAND[0]}..{BAND[1]}")
    print(f"flip residual sd in band: {flip_residual[band].std():.2f} steps\n")

    model, params = load_checkpoint(list_checkpoints(args.run)[-1][1])
    train_ok = train_targets.valid
    train_x = collect_site_features(model, params, probe_demos, train_idx, batch_size=args.batch_size)
    eval_x = collect_site_features(model, params, eval_demos, eval_idx, batch_size=args.batch_size)
    untrained = initial_params(model, jax.random.PRNGKey(1))
    train_u = collect_site_features(model, untrained, probe_demos, train_idx, batch_size=args.batch_size)
    eval_u = collect_site_features(model, untrained, eval_demos, eval_idx, batch_size=args.batch_size)

    depths = model.config.n_layers + 1
    layers: list[int | None] = [*range(depths), None]
    cells = cell_members(eval_targets.d_rich[band], eval_targets.d_poor[band], min_n=args.min_n)
    flip_centred, _ = centre_within_cells(flip_residual[band], cells)
    print(f"within-cell centring: {len(cells)} cells covering {sum(len(c) for c in cells):,} of the band\n")

    print("== 1. does the true-gap probe's residual track the model's flip residual? ==")
    print("Within a cell the true gap is constant, so the probe's variation is its residual;")
    print("corr is against the flip residual, centred the same way.")
    best, best_key = -np.inf, None
    for name, tx, ex in (("trained", train_x, eval_x), ("untrained", train_u, eval_u)):
        print(f"  {name}:")
        for site in SITES:
            row = []
            for layer in layers:
                result = fit_gap_probe(
                    flatten_depths(tx[site], layer)[train_ok],
                    train_targets.gap[train_ok].astype(np.float64),
                    flatten_depths(ex[site], layer)[band],
                    eval_targets.gap[band].astype(np.float64),
                )
                probe_centred, _ = centre_within_cells(result.predictions, cells)
                r = float(np.corrcoef(probe_centred, flip_centred)[0, 1])
                row.append(f"{r:+.3f}")
                if name == "trained" and result.r2 > best:
                    best, best_key = result.r2, (site, layer)
            print(f"    {site:<12} " + " ".join(f"{cell:>7}" for cell in row) + "   (layers 0..4, all)")
    print()

    site, layer = best_key
    print(f"== 2. target swap at {site} / {'all' if layer is None else f'layer {layer}'} (best true-gap calibration) ==")
    print("Half the bracketed band trains, half tests. Rows: what the probe was trained on;")
    print("columns: R2 against each target on the held-out half.")
    members = np.flatnonzero(band)
    order = np.random.default_rng(0).permutation(len(members))
    fit_half, test_half = members[order[::2]], members[order[1::2]]
    x = flatten_depths(eval_x[site], layer)
    targets = {"true gap": eval_targets.gap.astype(np.float64), "flip": flip}
    predictions = {}
    print(f"  {'trained on':<12} {'R2 true gap':>12} {'R2 flip':>9}")
    for train_name, y in targets.items():
        result = {}
        for eval_name, y_eval in targets.items():
            fitted = fit_gap_probe(x[fit_half], y[fit_half], x[test_half], y_eval[test_half])
            result[eval_name] = fitted.r2
            if eval_name == train_name:
                predictions[train_name] = fitted.predictions
        print(f"  {train_name:<12} {result['true gap']:>12.3f} {result['flip']:>9.3f}")
    r = float(np.corrcoef(predictions["true gap"], predictions["flip"])[0, 1])
    print(f"  corr(the two probes' held-out predictions) = {r:+.3f}")
    print()
    print("If the activations hold the model's own estimate, the flip-trained probe beats the")
    print("true-trained probe on flip; if they only hold the truth, the columns tie.")


if __name__ == "__main__":
    main()
