"""Summarise the offline value-axis results as a markdown table, per seed and across seeds.

    uv run python scripts/bc_value_axis_table.py [figures/data/bc]

Reads ``value_axis.<base>.<sweep>.json`` (027) and ``value_or_gap.<base>.json``
(028) and prints the numbers the handoff's summary table wants: competence of
the base, arm norms, drift and axis norms, collinearity, implied-offset slope,
split-half reliability, leave-one-out write error, slopes of exchange rate vs
offset, random-direction effect, cos(axis_0, axis_1) raw / disattenuated /
permutation p. Nothing is typed in here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def fmt(values, digits=2, signed=False):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return "-"
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    if len(values) == 1:
        return f.format(values[0])
    return f.format(np.mean(values)) + " ± " + f"{np.std(values, ddof=1):.{digits}f}"


def main() -> None:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "figures" / "data" / "bc"
    axes = [json.loads(p.read_text()) for p in sorted(data.glob("value_axis.*.json"))]
    gaps = [json.loads(p.read_text()) for p in sorted(data.glob("value_or_gap.*.json"))]
    if not axes:
        sys.exit(f"no value_axis.*.json under {data}")
    for sweep in sorted({a["sweep"] for a in axes}):
        rows = [a for a in axes if a["sweep"] == sweep]
        print(f"\n### sweep {sweep}  ({len(rows)} seeds: {', '.join(Path(r['run']).name for r in rows)})\n")
        print("| quantity | per seed | mean ± sd |")
        print("|---|---|---|")

        def line(label, key, digits=2, signed=False):
            vals = [key(r) for r in rows]
            print(f"| {label} | {' / '.join(fmt([v], digits, signed) for v in vals)} | {fmt(vals, digits, signed)} |")

        b = lambda r: r.get("behaviour", {})
        line("base chose_optimal (test, ρ=1)", lambda r: b(r).get("base", {}).get("chose_optimal"), 3)
        line("base exchange rate (steps; expert 10)", lambda r: b(r).get("base", {}).get("indifference"))
        line("null arm exchange rate", lambda r: b(r).get("null", {}).get("indifference"))
        line("‖Δθ‖ per arm, mean", lambda r: float(np.mean(r["arm_norms"])), 3)
        line("‖Δθ‖ per arm, sd", lambda r: float(np.std(r["arm_norms"], ddof=1)), 3)
        line("‖Δθ‖ null arm", lambda r: r.get("null_norm"), 3)
        line("‖drift‖", lambda r: r["drift_norm"], 3)
        line("‖axis‖ per unit value", lambda r: r["axis_norm"], 3)
        line("cos(drift, axis)", lambda r: r["cos_drift_axis"], 3, True)
        line("pairwise cos, same side, drift removed", lambda r: r["cos_same_residual"], 3, True)
        line("pairwise cos, opposite sides, drift removed", lambda r: r["cos_opposite_residual"], 3, True)
        line("implied offset vs trained: slope", lambda r: r["implied_slope"], 2, True)
        line("split-half reliability of the axis", lambda r: r["reliability"], 3, True)
        line("LOO write error, mean |written − arm| (steps)", lambda r: b(r).get("loo_error_mean_abs"))
        line("LOO write error with drift (steps)", lambda r: b(r).get("loo_error_mean_abs_with_drift"))
        line("slope exchange rate vs offset: arms", lambda r: b(r).get("slope_arms"), 1, True)
        line("slope: written (LOO)", lambda r: b(r).get("slope_written"), 1, True)
        line("slope: expert", lambda r: b(r).get("slope_expert"), 1, True)

        def arm_range(r):
            arms = b(r).get("arms", {})
            pts = [v["arm"]["indifference"] for v in arms.values() if np.isfinite(v["arm"]["indifference"])]
            return (max(pts) - min(pts)) if pts else None

        line("arms' exchange-rate range (steps)", arm_range, 1)

        def random_shift(r):
            c = b(r).get("controls", {})
            base = b(r).get("base", {}).get("indifference")
            vals = [abs(v["indifference"] - base) for k, v in c.items() if k.startswith("random_0.45") and np.isfinite(v["indifference"])]
            return float(np.mean(vals)) if vals else None

        line("random dir (|0.45·axis|): mean |shift| from base (steps)", random_shift)

        def random_reach(r):
            c = b(r).get("controls", {})
            vals = [v["reached"] for k, v in c.items() if k.startswith("random_0.45")]
            return float(np.mean(vals)) if vals else None

        line("random dir: reached", random_reach, 3)
        line("written +0.45 (LOO) reached", lambda r: (b(r).get("arms", {}).get("+0.45") or {}).get("written", {}).get("reached"), 3)
        line("written −0.45 (LOO) reached", lambda r: (b(r).get("arms", {}).get("-0.45") or {}).get("written", {}).get("reached"), 3)

    if gaps:
        print(f"\n### cos(axis_0, axis_1)  ({len(gaps)} seeds)\n")
        print("| quantity | per seed | mean ± sd |")
        print("|---|---|---|")
        for label, key, d, sg in (
            ("raw cos", lambda g: g["cos"], 3, True),
            ("reliability axis_0", lambda g: g["reliability_0"], 3, True),
            ("reliability axis_1", lambda g: g["reliability_1"], 3, True),
            ("disattenuated cos", lambda g: g["cos_disattenuated"], 3, True),
            ("permutation null mean", lambda g: g["null_mean"], 3, True),
            ("permutation null sd", lambda g: g["null_sd"], 3, False),
            ("p(null ≤ observed)", lambda g: g["p_permutation"], 4, False),
            ("cos(drift_0, drift_1)", lambda g: g["cos_drifts"], 3, True),
        ):
            vals = [key(g) for g in gaps]
            print(f"| {label} | {' / '.join(fmt([v], d, sg) for v in vals)} | {fmt(vals, d, sg)} |")


if __name__ == "__main__":
    main()
