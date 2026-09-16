"""One table per task from what the Craftax chains wrote: competence, proxy readout, axis.

    uv run python scripts/craftax_report.py figures/data/craftax/ore [--reference figures/data/bc]

Reads ``<base>.eval.csv`` (last row), ``value_axis.<base>.o{0,1}.json`` and
``value_or_gap.<base>.json`` in the directory. ``--reference`` prints the same
numbers for the maze BC bases beside them, which is the stage-2 competence
gate: the task is the maze's, so the engine-executed model must match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bases(directory: Path) -> list[str]:
    return sorted(p.name.replace(".eval.csv", "") for p in directory.glob("*.eval.csv"))


def final_row(directory: Path, base: str) -> dict:
    df = pd.read_csv(directory / f"{base}.eval.csv")
    return df.iloc[-1].to_dict()


def fmt(x, digits=1, pct=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "   -  "
    return f"{100 * x:5.{digits}f}%" if pct else f"{x:6.{digits}f}"


def competence(directory: Path, label: str) -> None:
    names = bases(directory)
    if not names:
        return
    print(f"\n{label}: final evaluation (held-out, values hidden)")
    print(
        f"{'base':16s} {'step':>7s} {'reached':>8s} {'optimal':>8s} {'legal':>8s} {'took f0 @1.0':>12s} {'@0.5':>7s} {'@0.0':>7s} {'indiff':>7s}"
    )
    for base in names:
        r = final_row(directory, base)
        print(
            f"{base:16s} {int(r['step']):7d} {fmt(r.get('rho100/reached')):>8s} {fmt(r.get('rho100/chose_optimal')):>8s} "
            f"{fmt(r.get('rho100/legal')):>8s} {fmt(r.get('rho100/followed_feature_zero')):>12s} "
            f"{fmt(r.get('rho050/followed_feature_zero')):>7s} {fmt(r.get('rho000/followed_feature_zero')):>7s} "
            f"{fmt(r.get('rho100/indifference'), 1, False):>7s}"
        )


def axis(directory: Path, label: str) -> None:
    files = sorted(directory.glob("value_axis.*.json"))
    if not files:
        return
    print(f"\n{label}: value axis (arms at 1k steps; exchange rate in actions)")
    print(
        f"{'base':16s} {'sweep':5s} {'base τ':>7s} {'slope arms':>10s} {'written':>8s} {'expert':>7s} {'LOO err':>8s} {'reliab':>7s} {'random':>7s}"
    )
    for path in files:
        d = json.loads(path.read_text())
        b = d["behaviour"]
        base_name = Path(d["run"]).name
        controls = b.get("controls", {})
        rand = (
            [v.get("indifference") for k, v in controls.items() if k.startswith("random")]
            if isinstance(controls, dict)
            else []
        )
        rand_shift = None
        if rand and b.get("base", {}).get("indifference") is not None:
            rand_shift = float(np.nanmean([abs(r - b["base"]["indifference"]) for r in rand if r is not None]))
        print(
            f"{base_name:16s} {d['sweep']:5s} {fmt(b.get('base', {}).get('indifference'), 1, False):>7s} "
            f"{fmt(b.get('slope_arms'), 1, False):>10s} {fmt(b.get('slope_written'), 1, False):>8s} {fmt(b.get('slope_expert'), 1, False):>7s} "
            f"{fmt(b.get('loo_error_mean_abs'), 1, False):>8s} {fmt(d.get('reliability'), 2, False):>7s} {fmt(rand_shift, 1, False):>7s}"
        )
    gaps = sorted(directory.glob("value_or_gap.*.json"))
    if gaps:
        print(f"\n{label}: one knob? cos(axis_0, axis_1)")
        for path in gaps:
            g = json.loads(path.read_text())
            print(
                f"{Path(g.get('run', path.stem)).name:16s} raw {g.get('cos', float('nan')):+.3f}  disattenuated {g.get('cos_disattenuated', float('nan')):+.3f}  "
                f"reliabilities {g.get('reliability_0', float('nan')):.2f}/{g.get('reliability_1', float('nan')):.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path, default=None, help="Maze BC results in the same layout, printed beside.")
    args = parser.parse_args()
    competence(args.directory, args.directory.name)
    if args.reference:
        competence(args.reference, f"reference {args.reference.name}")
    axis(args.directory, args.directory.name)
    if args.reference:
        axis(args.reference, f"reference {args.reference.name}")


if __name__ == "__main__":
    main()
