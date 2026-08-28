"""Steer the probed gap direction at SEP and ask what it does to the choice.

    uv run python scripts/steer_flip_bc.py /workspace/data/offline/runs/bcnv11.s1 \
        --flip figures/data/h1/flip/bcnv11.s1.npz > results/steer-flip-bcnv11.s1.txt

Three questions, sharing one set of steered decodes. The direction is the
true-gap ridge probe at SEP, injected at ``--depth`` (residual depths as in
``cell_residuals``: 0 = embedding, k = after block k) and calibrated so a dose
of +1 moves that probe's readout by one step.

1. **Dose-response.** If the direction is causally on the decision path,
   adding +d steps to the read gap shifts the psychometric crossing by -d:
   slope -1 means the model treats the written steps at full value. A
   norm-matched random direction is the control.
2. **Per-level flip doses.** Each level flips at some dose d*. If the steered
   quantity is the same one the behavioural flip points measure, then
   d* = theta - flip level by level: d* should track the flip point (within a
   cell, the flip residual) rather than the true gap.
3. **Clamp.** Overwrite the direction per level with the calibrated value for
   its true gap, removing the probe-visible residual. The within-cell choice
   variance that disappears is the share of xi that entered through this
   subspace - a dominance measurement with asymmetric force: sharpening is
   strong evidence, no sharpening is ambiguous while the probe reads only part
   of the gap. Shuffling the corrections within cells is the control (same
   edit sizes, wrong pairing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.analysis.flips import step_fit
from goalmisgen.analysis.probes import apply_linear, fit_ridge
from goalmisgen.offline.decode import replay_all
from goalmisgen.offline.demos import DemoSet, shared_levels
from goalmisgen.offline.gap_probe import cell_members, centre_within_cells, collect_site_features, gap_targets
from goalmisgen.offline.steer import steered_decode, unit_dose
from goalmisgen.offline.train import list_checkpoints, load_checkpoint
from goalmisgen.provenance import header

DEFAULT_PROBE_DEMOS = "/workspace/data/offline/demos/train.rho100"
DEFAULT_EVAL_DEMOS = "/workspace/data/offline/demos/test.rho100"
DOSES = (-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
BAND = (4, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--flip", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=3, help="Residual depth of the injection (and of the probe).")
    parser.add_argument(
        "--positions",
        choices=("sep", "prefix"),
        default="sep",
        help="Where to write: SEP alone, or every prefix position - the write a redundant readout cannot route around.",
    )
    parser.add_argument("--probe-demos", type=str, default=DEFAULT_PROBE_DEMOS)
    parser.add_argument("--eval-demos", type=str, default=DEFAULT_EVAL_DEMOS)
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--band", type=int, nargs=2, default=BAND, help="True-gap range the per-level statistics use.")
    return parser.parse_args()


def crossing(gap, took, level=0.5, min_n=25):
    grid = np.arange(-25, 60)
    rate = np.array([took[gap == g].mean() if (gap == g).sum() >= min_n else np.nan for g in grid])
    good = np.isfinite(rate)
    xs, ys = grid[good], rate[good]
    for i in range(len(xs) - 1):
        if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
            return float(xs[i] + (ys[i] - level) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i]))
    return float("nan")


def fit_direction(x, y, seed=0, l2_grid=(1e-2, 1e-1, 1.0, 10.0, 100.0)):
    order = np.random.default_rng(seed).permutation(len(x))
    cut = max(1, len(order) // 5)
    val, fit = order[:cut], order[cut:]

    def val_mse(l2):
        w, mean, std = fit_ridge(x[fit], y[fit], l2=l2)
        return float(np.mean((apply_linear(x[val], w, mean, std) - y[val]) ** 2))

    best = min(l2_grid, key=val_mse)
    return fit_ridge(x, y, l2=best)


def main() -> None:
    args = parse_args()
    print(header())
    print()

    flip_data = np.load(args.flip)
    flip = flip_data["flip"]
    n_eval = len(flip)

    probe_demos = DemoSet.load(args.probe_demos, hide_values=True)
    eval_demos = DemoSet.load(args.eval_demos, hide_values=True)
    if shared_levels(probe_demos, eval_demos):
        raise SystemExit("probe and eval demos share levels")
    model, params = load_checkpoint(list_checkpoints(args.run)[-1][1])
    if not 0 <= args.depth <= model.config.n_layers:
        raise SystemExit(f"--depth {args.depth} out of range for a {model.config.n_layers}-block model")

    train_idx = np.arange(min(args.n_train, len(probe_demos)))
    eval_idx = np.arange(n_eval)
    train_targets = gap_targets(probe_demos, train_idx)
    eval_targets = gap_targets(eval_demos, eval_idx)
    if not np.array_equal(flip_data["gap"], eval_targets.gap):
        raise SystemExit("flip file's gaps disagree with the demonstration set")
    gap = eval_targets.gap

    # A prefix-wide write adds the same vector to every cell, which moves the
    # cell-mean readout by that vector - so the cells_mean probe calibrates it.
    site = "sep" if args.positions == "sep" else "cells_mean"
    train_x = collect_site_features(model, params, probe_demos, train_idx)[site][:, args.depth]
    eval_x = collect_site_features(model, params, eval_demos, eval_idx)[site][:, args.depth]
    w, mean, std = fit_direction(train_x[train_targets.valid], train_targets.gap[train_targets.valid].astype(np.float64))
    readout = apply_linear(eval_x, w, mean, std)
    u = unit_dose(w, std)
    probe_r2 = 1 - np.var(readout[eval_targets.valid] - gap[eval_targets.valid]) / np.var(gap[eval_targets.valid])
    print(
        f"direction: true-gap probe at {site} depth {args.depth}, written at {args.positions};"
        f"  eval r2 {probe_r2:+.3f};  |unit dose| {np.linalg.norm(u):.3f}"
    )

    observations = eval_demos.observations(eval_idx)

    def decode_with(delta) -> tuple[np.ndarray, np.ndarray]:
        decoded = steered_decode(model, params, observations, args.depth, delta, positions=args.positions)
        outcomes = replay_all(eval_demos, eval_idx, decoded)
        reached = np.array([bool(o.get("reached_objective")) for o in outcomes])
        fid = np.array([-1 if o.get("reached_feature_id") is None else int(o["reached_feature_id"]) for o in outcomes])
        colour = np.asarray(eval_demos.feature_ids)[eval_idx][np.arange(n_eval), eval_targets.richer]
        return fid == colour, reached

    print("\n== 1. dose-response: does writing +d steps of gap shift the crossing by -d? ==")
    took_by_dose, reached_by_dose = {}, {}
    for dose in DOSES:
        took, reached = decode_with(np.tile(dose * u, (n_eval, 1)).astype(np.float32))
        took_by_dose[dose], reached_by_dose[dose] = took, reached
        ok = reached & eval_targets.valid
        print(f"  dose {dose:+5.1f}: crossing {crossing(gap[ok], took[ok].astype(float)):6.2f}   reached {reached.mean():.1%}")
    theta0 = crossing(gap[reached_by_dose[0.0] & eval_targets.valid], took_by_dose[0.0][reached_by_dose[0.0] & eval_targets.valid].astype(float))
    small = [d for d in DOSES if abs(d) <= 3]
    xs = np.array(small)
    ys = np.array([crossing(gap[reached_by_dose[d] & eval_targets.valid], took_by_dose[d][reached_by_dose[d] & eval_targets.valid].astype(float)) for d in small])
    keep = np.isfinite(ys)
    if keep.sum() >= 2:
        slope = np.polyfit(xs[keep], ys[keep], 1)[0]
        print(f"  slope over |dose| <= 3: {slope:+.3f}   (fully causal and calibrated would be -1)")
    else:
        print("  slope over |dose| <= 3: not enough finite crossings")

    rng = np.random.default_rng(0)
    random_direction = rng.normal(size=u.shape)
    random_direction *= np.linalg.norm(u) / np.linalg.norm(random_direction)
    print("  norm-matched random direction:")
    for dose in (-8.0, -4.0, 4.0, 8.0):
        took, reached = decode_with(np.tile(dose * random_direction, (n_eval, 1)).astype(np.float32))
        ok = reached & eval_targets.valid
        print(f"    dose {dose:+5.1f}: crossing {crossing(gap[ok], took[ok].astype(float)):6.2f}")

    print("\n== 2. per-level flip doses: do they land where the behavioural flip points say? ==")
    lo, hi = args.band
    band = np.isfinite(flip) & (gap >= lo) & (gap <= hi) & eval_targets.valid
    pseudo_theta = np.array(sorted(-d for d in DOSES), dtype=float)
    chosen = np.stack([took_by_dose[d] for d in sorted(DOSES, reverse=True)])[:, band]
    doses_fit = step_fit(pseudo_theta, chosen)
    d_star = -doses_fit.flip  # taken exactly for doses below d*
    both = np.isfinite(d_star)
    print(f"  {both.sum():,} of {band.sum():,} band levels flip within the dose grid")
    if both.sum() >= 10:
        fl, tg = flip[band][both], gap[band][both]
        print(f"  violations mean {doses_fit.violations[both].mean():.2f} of {len(DOSES)}")
        print(f"  corr(d*, theta0 - flip) = {np.corrcoef(d_star[both], theta0 - fl)[0, 1]:+.3f}     slope of d* on flip {np.polyfit(fl, d_star[both], 1)[0]:+.3f} (predicted -1)")
        print(f"  corr(d*, theta0 - gap)  = {np.corrcoef(d_star[both], theta0 - tg)[0, 1]:+.3f}")
        cells = cell_members(eval_targets.d_rich[band][both], eval_targets.d_poor[band][both], min_n=args.min_n)
        if cells:
            a, _ = centre_within_cells(d_star[both], cells)
            b, _ = centre_within_cells(fl, cells)
            print(f"  within-cell corr(d*, -flip residual) = {np.corrcoef(a, -b)[0, 1]:+.3f}   ({len(cells)} cells)")
            print("  Within a cell the true gap is constant, so only the model's own misread can carry this.")
    else:
        print("  too few flips for the per-level statistics (a direction with no causal effect looks like this)")

    print("\n== 3. clamp: remove the probe-visible residual, does the curve sharpen? ==")
    valid = eval_targets.valid
    a_fit = np.polyfit(gap[valid], readout[valid], 1)
    correction = (np.polyval(a_fit, gap) - readout).astype(np.float32)
    correction[~valid] = 0.0

    def within_cell_stats(took, reached):
        ok = reached & valid & (gap >= lo) & (gap <= hi)
        cells = cell_members(eval_targets.d_rich[ok], eval_targets.d_poor[ok], min_n=args.min_n)
        t = took[ok]
        variance = sum(len(m) * t[m].mean() * (1 - t[m].mean()) for m in cells) / sum(len(m) for m in cells)
        okc = reached & valid
        width = crossing(gap[okc], took[okc].astype(float), 0.25) - crossing(gap[okc], took[okc].astype(float), 0.75)
        return variance, width

    base_var, base_width = within_cell_stats(took_by_dose[0.0], reached_by_dose[0.0])
    took_c, reached_c = decode_with(correction[:, None] * u[None, :])
    clamp_var, clamp_width = within_cell_stats(took_c, reached_c)
    shuffled = correction.copy()
    for members in cell_members(eval_targets.d_rich, eval_targets.d_poor, min_n=2):
        shuffled[members] = shuffled[members][rng.permutation(len(members))]
    took_s, reached_s = decode_with(shuffled[:, None] * u[None, :])
    shuf_var, shuf_width = within_cell_stats(took_s, reached_s)
    print(f"  {'':<18} {'within-cell var':>16} {'curve width':>12}")
    print(f"  {'base':<18} {base_var:>16.4f} {base_width:>12.2f}")
    print(f"  {'clamped':<18} {clamp_var:>16.4f} {clamp_width:>12.2f}")
    print(f"  {'shuffled control':<18} {shuf_var:>16.4f} {shuf_width:>12.2f}")
    print(f"  variance removed by clamping: {1 - clamp_var / base_var:+.1%}  (control {1 - shuf_var / base_var:+.1%})")
    print("  The removed share is xi that entered through this subspace; no sharpening is")
    print("  ambiguous while the probe reads the gap at r2 ~ 0.3-0.5, sharpening is not.")


if __name__ == "__main__":
    main()
