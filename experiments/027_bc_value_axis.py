"""Is what an objective is worth, in the route model, a direction in weight space?

    uv run python experiments/027_bc_value_axis.py /workspace/data/offline/runs/bcnv11.s1 \\
        --sweep o0 --steps 1000 --demos /workspace/data/offline/demos/test.rho100 [--json out.json]

The offline twin of ``014_value_axis_analysis.py``. The base is a route model
trained *without the value channel* (``bcnv11.s*``), so the values (1.0, 0.5)
are learned constants and a fine-tune onto other values has to move weights.
The arms are that base fine-tuned for one fixed budget on demonstrations at
shifted values of one objective (``scripts/value_axis_arms.py``); each arm's
diff from the base is one sample of "what changing the value by ``offset``
does to the weights".

Same questions, same arithmetic (``goalmisgen.analysis.weights``):

``collinear``    with the common component (drift) removed, arms on the same
                 side agree and arms on opposite sides oppose
``graded``       an axis fitted without an arm reads back that arm's offset
``writable``     base + offset * axis_{-i}, decoded greedily on held-out
                 levels at the base values, reproduces arm i's exchange rate;
                 norm-matched random directions do nothing; the null arm (same
                 budget, same values) only drifts

The null arm is held out of the fit and reported against it. Behaviour is the
greedy route replayed under the environment's rules, on the same held-out
levels for every parameter vector, so the exchange rate (``indifference``: the
distance gap at which colour 0 is taken half the time) is read identically for
arms, writes and controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from goalmisgen.analysis.weights import cosine, explained, fit_axis_and_drift, projected_offset
from goalmisgen.offline.axis import (
    arm_dirs,
    arm_params,
    expected_indifference,
    load_base,
    load_diffs,
    measure,
    measure_flat,
)
from goalmisgen.offline.demos import DemoSet
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Base run directory; arms under <run>/arms/.")
    parser.add_argument("--sweep", default="o0", help="o0 moves colour 0's value, o1 colour 1's.")
    parser.add_argument("--steps", type=int, default=1000, help="Arm budget, as in the arm directory names.")
    parser.add_argument("--demos", type=Path, required=True, help="Held-out demonstrations at the base values (test split).")
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument("--extrapolate", type=float, nargs="*", default=[0.6, -0.6, 0.9, -0.9], help="Offsets outside the grid to write with the full axis.")
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    base = load_base(args.run)
    objective = int(args.sweep[-1])
    print(f"base {args.run.name} @ step {base.step:,}; {base.flat.size:,} parameters; values {base.values}; values {'hidden' if base.hide_values else 'shown'}")
    arms = arm_dirs(args.run, args.sweep, args.steps)
    if not arms:
        sys.exit(f"no finished arms of sweep {args.sweep} at {args.steps} steps under {args.run / 'arms'}")
    diffs = load_diffs(base, arms)
    null = diffs.get(0.0)
    fitted = {o: d for o, d in diffs.items() if abs(o) > 1e-9}
    offsets = np.array(sorted(fitted))
    stacked = np.stack([fitted[o] for o in offsets])
    if len(offsets) < 3:
        sys.exit("need at least three arms away from the base before any of this means anything")
    axis, drift = fit_axis_and_drift(offsets, stacked)
    norms = np.array([np.linalg.norm(fitted[o]) for o in offsets])
    print(f"sweep {args.sweep}: {len(offsets)} arms at offsets {offsets.min():+.2f}..{offsets.max():+.2f}" + (" + null arm" if null is not None else " (no null arm)"))
    print(f"|delta theta| per arm  {norms.mean():.4g} +- {norms.std(ddof=1):.4g}  (min {norms.min():.4g}, max {norms.max():.4g})")
    print(f"|drift| {np.linalg.norm(drift):.4g}   |axis| per unit value {np.linalg.norm(axis):.4g}   cos(drift, axis) {cosine(drift, axis):+.3f}")
    if null is not None:
        print(f"null arm |delta| {np.linalg.norm(null):.4g}, cos(null, drift) {cosine(null, drift):+.3f}, cos(null - drift, axis) {cosine(null - drift, axis):+.3f}")

    # --- collinear? ---------------------------------------------------------
    residual = {o: fitted[o] - drift for o in offsets}
    same_raw, opp_raw, same_res, opp_res = [], [], [], []
    for i, a in enumerate(offsets):
        for b in offsets[i + 1 :]:
            raw, res = cosine(fitted[a], fitted[b]), cosine(residual[a], residual[b])
            (same_raw if a * b > 0 else opp_raw).append(raw)
            (same_res if a * b > 0 else opp_res).append(res)
    print("\n=== collinear? mean pairwise cosine between arms ===")
    print(f"  {'':>28}{'same side':>12}{'opposite':>12}")
    print(f"  {'raw diffs':>28}{np.mean(same_raw):>+12.3f}{np.mean(opp_raw):>+12.3f}")
    print(f"  {'common component removed':>28}{np.mean(same_res):>+12.3f}{np.mean(opp_res):>+12.3f}")
    print("  Raw positive everywhere = 'was fine-tuned'. With drift removed, same side should agree and opposite sides oppose.")

    # --- graded and predictive? --------------------------------------------
    print("\n=== graded and predictive? leave-one-out fit ===")
    print(f"  {'offset':>8}{'|delta|':>10}{'|resid|':>10}{'held-out R2':>13}{'held-out cos':>14}{'implied':>10}")
    loo = {}
    for o in offsets:
        others = [p for p in offsets if p != o]
        held_axis, held_drift = fit_axis_and_drift(np.array(others), np.stack([fitted[p] for p in others]))
        left = fitted[o] - held_drift
        loo[o] = dict(axis=held_axis, drift=held_drift, r2=explained(left, o, held_axis), cos=cosine(left, held_axis), implied=projected_offset(left, held_axis))
        print(f"  {o:>+8.2f}{np.linalg.norm(fitted[o]):>10.4g}{np.linalg.norm(residual[o]):>10.4g}{loo[o]['r2']:>13.3f}{loo[o]['cos']:>+14.3f}{loo[o]['implied']:>+10.2f}")
    if null is not None:
        print(f"  {'null':>8}{np.linalg.norm(null):>10.4g}{np.linalg.norm(null - drift):>10.4g}{'-':>13}{cosine(null - drift, axis):>+14.3f}{projected_offset(null - drift, axis):>+10.2f}")
    implied = np.array([loo[o]["implied"] for o in offsets])
    slope = float(np.polyfit(offsets, implied, 1)[0])
    print(f"  implied offset vs trained offset: correlation {np.corrcoef(offsets, implied)[0, 1]:+.3f}, slope {slope:.2f} (1 = the axis reads the offset back at scale)")

    # split-half reliability: two independent fits of the same axis
    halves = (offsets[0::2], offsets[1::2])
    half_axes = [fit_axis_and_drift(np.array(h), np.stack([fitted[p] for p in h]))[0] for h in halves]
    reliability = cosine(*half_axes)
    print(f"  split-half reliability of the axis {reliability:+.3f}")

    out = {
        "run": str(args.run), "sweep": args.sweep, "steps": args.steps, "base_step": base.step,
        "parameters": int(base.flat.size), "offsets": offsets.tolist(),
        "arm_norms": norms.tolist(), "drift_norm": float(np.linalg.norm(drift)), "axis_norm": float(np.linalg.norm(axis)),
        "cos_drift_axis": cosine(drift, axis),
        "null_norm": None if null is None else float(np.linalg.norm(null)),
        "cos_same_raw": float(np.mean(same_raw)), "cos_opposite_raw": float(np.mean(opp_raw)),
        "cos_same_residual": float(np.mean(same_res)), "cos_opposite_residual": float(np.mean(opp_res)),
        "loo": {f"{o:+.2f}": {k: float(v) for k, v in loo[o].items() if k in ("r2", "cos", "implied")} for o in offsets},
        "implied_slope": slope, "reliability": reliability,
        "behaviour": {},
    }

    if args.skip_behaviour:
        finish(args, out)
        return

    # --- writable? -----------------------------------------------------------
    demos = DemoSet.load(args.demos, hide_values=base.hide_values)
    indices = np.arange(min(args.levels, len(demos)))
    def say(label, m, expected=None):
        e = "" if expected is None else f"{expected:>9.1f}"
        print(f"  {label:<34}{m.indifference:>9.2f}{e:>9}{m.chose_optimal:>10.3f}{m.reached:>9.3f}{m.legal:>8.3f}{m.followed_f0:>8.3f}")

    print(f"\n=== writable? greedy decode on {len(indices)} held-out levels at the base values ===")
    print(f"  {'':<34}{'indiff.':>9}{'expert':>9}{'optimal':>10}{'reached':>9}{'legal':>8}{'f0':>8}")
    base_m = measure(base, base.params, demos, indices)
    say("base, untouched", base_m, expected_indifference(base.values, objective, 0.0))
    out["behaviour"]["base"] = base_m.as_row()
    if null is not None:
        null_m = measure(base, arm_params(arms[0.0]), demos, indices)
        say("null arm (fine-tuned at base values)", null_m, expected_indifference(base.values, objective, 0.0))
        out["behaviour"]["null"] = null_m.as_row()

    print("\n  each offset: the arm itself | written from an axis fitted WITHOUT it (base + o*axis) | the same plus drift")
    print(f"  {'offset':>8}{'expert':>8}{'arm':>8}{'written':>9}{'+drift':>9}{'| reach arm':>11}{'writ':>7}{'+drift':>7}{'| optimal arm':>13}{'writ':>7}{'+drift':>7}")
    per_arm = {}
    errors, errors_drift = [], []
    for o in offsets:
        arm_m = measure(base, arm_params(arms[o]), demos, indices)
        written_m = measure_flat(base, base.flat + o * loo[o]["axis"], demos, indices)
        with_drift_m = measure_flat(base, base.flat + loo[o]["drift"] + o * loo[o]["axis"], demos, indices)
        expected = expected_indifference(base.values, objective, float(o))
        per_arm[f"{o:+.2f}"] = {"expected": expected, "arm": arm_m.as_row(), "written": written_m.as_row(), "written_with_drift": with_drift_m.as_row()}
        errors.append(written_m.indifference - arm_m.indifference)
        errors_drift.append(with_drift_m.indifference - arm_m.indifference)
        print(
            f"  {o:>+8.2f}{expected:>8.1f}{arm_m.indifference:>8.2f}{written_m.indifference:>9.2f}{with_drift_m.indifference:>9.2f}"
            f"{arm_m.reached:>11.3f}{written_m.reached:>7.3f}{with_drift_m.reached:>7.3f}"
            f"{arm_m.chose_optimal:>13.3f}{written_m.chose_optimal:>7.3f}{with_drift_m.chose_optimal:>7.3f}"
        )
    out["behaviour"]["arms"] = per_arm
    errors, errors_drift = np.array(errors), np.array(errors_drift)
    arm_points = np.array([per_arm[f"{o:+.2f}"]["arm"]["indifference"] for o in offsets])
    finite = np.isfinite(errors) & np.isfinite(arm_points)
    lo, hi = expected_indifference(base.values, objective, float(offsets.min())), expected_indifference(base.values, objective, float(offsets.max()))
    print(f"\n  arms' exchange rates span {np.nanmin(arm_points):.1f}..{np.nanmax(arm_points):.1f} steps (expert {min(lo, hi):.1f}..{max(lo, hi):.1f})")
    print(f"  leave-one-out write error, mean |written - arm|: {np.nanmean(np.abs(errors[finite])):.2f} steps (signed mean {np.nanmean(errors[finite]):+.2f}); with drift {np.nanmean(np.abs(errors_drift[finite])):.2f} ({np.nanmean(errors_drift[finite]):+.2f})")
    slope_arm = float(np.polyfit(offsets[finite], arm_points[finite], 1)[0])
    written_points = np.array([per_arm[f"{o:+.2f}"]["written"]["indifference"] for o in offsets])
    ok = np.isfinite(written_points)
    slope_written = float(np.polyfit(offsets[ok], written_points[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    slope_expert = 20.0 if objective == 0 else -20.0
    print(f"  slope of exchange rate vs offset: arms {slope_arm:+.1f}, written {slope_written:+.1f}, expert {slope_expert:+.1f} steps per unit value")
    out["behaviour"]["loo_error_mean_abs"] = float(np.nanmean(np.abs(errors[finite])))
    out["behaviour"]["loo_error_mean_abs_with_drift"] = float(np.nanmean(np.abs(errors_drift[finite])))
    out["behaviour"]["slope_arms"], out["behaviour"]["slope_written"], out["behaviour"]["slope_expert"] = slope_arm, slope_written, slope_expert

    print("\n  controls and extrapolation (full axis)")
    print(f"  {'':<34}{'indiff.':>9}{'expert':>9}{'optimal':>10}{'reached':>9}{'legal':>8}{'f0':>8}")
    rng = np.random.default_rng(args.seed)
    controls = {}
    for magnitude in (0.45, 0.2):
        for draw in range(args.random_draws):
            direction = rng.normal(size=axis.size)
            direction *= np.linalg.norm(magnitude * axis) / np.linalg.norm(direction)
            m = measure_flat(base, base.flat + direction, demos, indices)
            say(f"random, |.| = |{magnitude:.2f} axis| #{draw}", m)
            controls[f"random_{magnitude:.2f}_{draw}"] = m.as_row()
    m = measure_flat(base, base.flat + drift, demos, indices)
    say("base + drift only", m)
    controls["drift_only"] = m.as_row()
    for o in args.extrapolate:
        m = measure_flat(base, base.flat + o * axis, demos, indices)
        say(f"written {o:+.2f} (outside the grid)", m, expected_indifference(base.values, objective, o))
        controls[f"extrapolate_{o:+.2f}"] = m.as_row()
    out["behaviour"]["controls"] = controls
    print(
        "\nA direction is an axis if the written column tracks the arm column arm by arm,\n"
        "random directions of the same length leave the base where it was, and the null\n"
        "arm only drifts. Absolute levels may sit off the expert's (the base's own\n"
        "exchange rate does too); the slope is the reading."
    )
    finish(args, out)


def finish(args, out):
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
