"""How many value axes are there, and is writing one of them in one place enough?

    uv run python experiments/029_axis_shape.py /workspace/data/offline/runs/bcnv11.s1 \\
        --sweep o0 --steps 1000 --demos /workspace/data/offline/demos/test.rho100 [--json out.json]

``027`` asks whether the arms of a sweep collapse to a line and whether that line
can be written to. It answers both with one number each, and neither number can
fail in the two ways the width/depth campaign expects.

**How many axes.** A leave-one-out fit reports that the axis explains a lot; it
never reports what is left. The task has one goal degree of freedom -- ``015``
and ``028`` established the knob is the *gap* between two objective values, not
the values -- so the true rank is one and a second direction that replicates
across disjoint halves of the arms is something the network did. That is the
rarest thing about this setup and the first half of what is measured here.

**How many places.** A share of ``||axis||^2`` in a module is not evidence that
writing there does anything. The two come apart exactly where depth is supposed
to break the account: a quantity recomputed at every layer can have its axis
spread thin and still resist being written one layer at a time. So the axis is
restricted to a group, or to the largest ``k`` groups, written into the base's
own weights, and the *behaviour* is read -- the same greedy decode on the same
held-out levels ``027`` uses, so the exchange rates are directly comparable.

What would refute the campaign's registered predictions, in the terms of
``Preregistration-scaling.md``: a residual direction replicating above 0.50 (P3),
or a single block reaching within the tolerance of the full-axis write at
``L = 16`` (P4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from goalmisgen.analysis.layers import (
    axis_by_group,
    blocks_to_cover,
    group_shares,
    parameter_groups,
    restrict,
)
from goalmisgen.analysis.spectrum import (
    axis_removed_operator,
    drift_removed_operator,
    gram_matrix,
    participation_ratio,
    permutation_participation_ratio,
    residual_reliability,
    spectrum,
    variance_shares,
)
from goalmisgen.analysis.weights import fit_axis_and_drift
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
    parser.add_argument("--demos", type=Path, required=True, help="Held-out demonstrations at the base values.")
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Steps of exchange rate within which a partial write counts as reaching the full one.",
    )
    parser.add_argument("--splits", type=int, default=200, help="Random half-splits per reliability estimate.")
    parser.add_argument("--resamples", type=int, default=1000, help="Offset shuffles for the participation-ratio null.")
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
    arms = arm_dirs(args.run, args.sweep, args.steps)
    if not arms:
        sys.exit(f"no finished arms of sweep {args.sweep} at {args.steps} steps under {args.run / 'arms'}")
    diffs = load_diffs(base, arms)
    fitted = {o: d for o, d in diffs.items() if abs(o) > 1e-9}
    offsets = np.array(sorted(fitted))
    if len(offsets) < 3:
        sys.exit("need at least three arms away from the base before any of this means anything")
    stacked = np.stack([fitted[o] for o in offsets])
    axis, drift = fit_axis_and_drift(offsets, stacked)

    print(
        f"base {args.run.name} @ step {base.step:,}; {base.flat.size:,} parameters; "
        f"values {base.values}; values {'hidden' if base.hide_values else 'shown'}"
    )
    print(f"sweep {args.sweep}: {len(offsets)} arms at offsets {offsets.min():+.2f}..{offsets.max():+.2f}")

    out = {
        "run": str(args.run),
        "sweep": args.sweep,
        "steps": args.steps,
        "base_step": base.step,
        "parameters": int(base.flat.size),
        "offsets": offsets.tolist(),
    }

    # --- how many axes? ------------------------------------------------------
    gram = gram_matrix(stacked)
    without_drift = drift_removed_operator(offsets)
    without_axis = axis_removed_operator(offsets)
    family = spectrum(without_drift @ gram @ without_drift.T)
    residual = spectrum(without_axis @ gram @ without_axis.T)
    family_pr = participation_ratio(family)
    residual_pr = participation_ratio(residual)
    null_pr = permutation_participation_ratio(offsets, gram, resamples=args.resamples, seed=args.seed)
    replication = residual_reliability(offsets, gram, splits=args.splits, seed=args.seed)
    shares = variance_shares(family)

    print("\n=== how many axes? spectrum of the arms with the common component removed ===")
    print("  variance share by direction: " + "  ".join(f"{s:.3f}" for s in shares[: min(6, len(shares))]))
    print(f"  participation ratio {family_pr:.2f}   (1.00 = one direction carries the family)")
    print(
        f"  after the fitted axis is also removed: {residual_pr:.2f}"
        f"   permuted-offset null {null_pr.mean():.2f} [{np.percentile(null_pr, 5):.2f}, {np.percentile(null_pr, 95):.2f}]"
    )
    print(f"  leading residual direction, split-half replication {replication:.3f}")
    print(
        "  The task has one goal degree of freedom, so rank one is the correct answer and a\n"
        "  replicating residual direction is the network's doing. Read the replication figure,\n"
        "  not the participation ratio: fine-tuning noise inflates the ratio by an amount that\n"
        "  is not comparable across network shapes, which is what the permuted null calibrates."
    )
    out["spectrum"] = {
        "variance_shares": shares.tolist(),
        "participation_ratio": family_pr,
        "residual_participation_ratio": residual_pr,
        "null_participation_ratio_mean": float(null_pr.mean()),
        "null_participation_ratio_p05": float(np.percentile(null_pr, 5)),
        "null_participation_ratio_p95": float(np.percentile(null_pr, 95)),
        "residual_reliability": replication,
    }

    # --- how many places? ----------------------------------------------------
    groups = parameter_groups(base.params)
    profile = axis_by_group(offsets, stacked, groups, splits=args.splits, seed=args.seed)
    shares_by_group = group_shares(axis, groups)
    order = sorted(shares_by_group, key=shares_by_group.get, reverse=True)
    cover = blocks_to_cover(shares_by_group, 0.9)

    print("\n=== where does it live? the axis refitted inside each module alone ===")
    print(f"  {'module':<20}{'params':>12}{'share':>9}{'of params':>11}{'enrichment':>12}{'reliability':>13}")
    for name in order:
        row = profile[name]
        print(
            f"  {name:<20}{int(row['parameters']):>12,}{row['share']:>9.3f}"
            f"{row['parameter_share']:>11.3f}{row['enrichment']:>12.2f}{row['reliability']:>13.3f}"
        )
    print(f"  modules holding 90% of ||axis||^2: {cover} of {len(groups)}")
    print(
        "  Enrichment, not share: a module holding 40% of the weights takes about 40% of a\n"
        "  random direction. Localisation is not storage -- where a diff lands is where the\n"
        "  gradient found it cheapest to write."
    )
    out["groups"] = profile
    out["blocks_to_cover_90"] = cover

    if args.skip_behaviour:
        finish(args, out)
        return

    # --- is one place enough? ------------------------------------------------
    demos = DemoSet.load(args.demos, hide_values=base.hide_values)
    indices = np.arange(min(args.levels, len(demos)))
    extremes = [float(offsets.min()), float(offsets.max())]

    print(f"\n=== is one place enough? greedy decode on {len(indices)} held-out levels ===")
    print(f"  {'':<26}{'offset':>8}{'indiff.':>9}{'expert':>8}{'reached':>9}{'optimal':>9}")
    base_m = measure(base, base.params, demos, indices)
    print(
        f"  {'base, untouched':<26}{'-':>8}{base_m.indifference:>9.2f}{'-':>8}{base_m.reached:>9.3f}{base_m.chose_optimal:>9.3f}"
    )
    out["behaviour"] = {"base": base_m.as_row(), "extremes": {}}

    for offset in extremes:
        others = [p for p in offsets if p != offset]
        held_axis, _ = fit_axis_and_drift(np.array(others), np.stack([fitted[p] for p in others]))
        arm_m = measure(base, arm_params(arms[offset]), demos, indices)
        full_m = measure_flat(base, base.flat + offset * held_axis, demos, indices)
        expected = expected_indifference(base.values, objective, offset)
        entry = {"expected": expected, "arm": arm_m.as_row(), "full_axis": full_m.as_row(), "single": {}, "cumulative": []}
        print()
        print(
            f"  {'the arm itself':<26}{offset:>+8.2f}{arm_m.indifference:>9.2f}{expected:>8.1f}{arm_m.reached:>9.3f}{arm_m.chose_optimal:>9.3f}"
        )
        print(
            f"  {'full axis, written':<26}{offset:>+8.2f}{full_m.indifference:>9.2f}{expected:>8.1f}{full_m.reached:>9.3f}{full_m.chose_optimal:>9.3f}"
        )

        for name in order:
            part = measure_flat(base, base.flat + offset * restrict(held_axis, groups, name), demos, indices)
            entry["single"][name] = part.as_row()
            gap = abs(part.indifference - full_m.indifference)
            flag = " <- reaches the full write" if gap <= args.tolerance else ""
            print(
                f"  {'only ' + name:<26}{offset:>+8.2f}{part.indifference:>9.2f}{expected:>8.1f}"
                f"{part.reached:>9.3f}{part.chose_optimal:>9.3f}{flag}"
            )

        needed = None
        for count in range(1, len(order) + 1):
            part = measure_flat(base, base.flat + offset * restrict(held_axis, groups, order[:count]), demos, indices)
            entry["cumulative"].append({"modules": order[:count], "behaviour": part.as_row()})
            if needed is None and abs(part.indifference - full_m.indifference) <= args.tolerance:
                needed = count
                print(
                    f"  {'largest ' + str(count) + ' modules':<26}{offset:>+8.2f}{part.indifference:>9.2f}"
                    f"{expected:>8.1f}{part.reached:>9.3f}{part.chose_optimal:>9.3f} <- within tolerance"
                )
                break
        entry["modules_needed"] = needed
        if needed is None:
            print(f"  no subset of modules reached the full write within {args.tolerance} steps")
        out["behaviour"]["extremes"][f"{offset:+.2f}"] = entry

    needed = [e["modules_needed"] for e in out["behaviour"]["extremes"].values() if e["modules_needed"]]
    if needed:
        print(f"\n  modules needed to reach the full write within {args.tolerance} steps: {max(needed)} of {len(groups)}")
    finish(args, out)


def finish(args: argparse.Namespace, out: dict) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
