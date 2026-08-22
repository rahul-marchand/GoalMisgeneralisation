"""Which fine-tuning learning rate to give one cell of the width/depth grid.

    uv run python scripts/pick_arm_lr.py --candidates RUN [RUN ...] --target 19.0

The same learning rate is a different-sized step at a different ``d_model``, so a
grid of shapes fine-tuned at one fixed rate would report "the wide models moved
less" as "the wide models have a cleaner axis". Each cell therefore trains its
*widest positive arm* at a small ladder of rates, and this picks the one whose
achieved exchange rate lands closest to that arm's expert target.

One scalar per cell, read off one arm's behaviour. It cannot reach the axis,
which is fitted from how a cell's arms differ from *each other* at a fixed
budget -- see the 2026-08-22 amendment in ``Preregistration-scaling.md`` for why
the budget is fixed rather than itself behavioural.

Reads ``eval.csv``, so it needs the arms to have been trained with
``--eval base=<the base's test demonstrations>``: that column is the exchange
rate measured the way ``027`` measures it, at the base values, which is the
quantity the campaign reports. Prints the chosen rate on stdout and the ladder
on stderr, so a shell can capture one and a human can read the other.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def final_row(run: Path, column: str) -> float:
    rows = list(csv.DictReader((run / "eval.csv").open()))
    if not rows:
        raise SystemExit(f"{run}/eval.csv has no rows; was the arm trained with --eval?")
    if column not in rows[-1]:
        raise SystemExit(f"{run}/eval.csv has no column {column!r}; it has {sorted(rows[-1])}")
    return float(rows[-1][column])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True, help="Calibration arm run directories.")
    parser.add_argument("--target", type=float, required=True, help="The arm's expert exchange rate, in steps.")
    parser.add_argument("--column", default="base/indifference", help="Which eval.csv column holds the exchange rate.")
    args = parser.parse_args()

    ladder = []
    for run in args.candidates:
        if not (run / "done.json").exists():
            print(f"skipping {run}: not finished", file=sys.stderr)
            continue
        rate = float(json.loads((run / "config.json").read_text())["train"]["learning_rate"])
        reached = final_row(run, args.column)
        ladder.append((abs(reached - args.target), rate, reached, run))
    if not ladder:
        raise SystemExit("no finished calibration arms")

    ladder.sort()
    print(f"{'lr':>10}{'reached':>10}{'target':>9}{'|error|':>9}  run", file=sys.stderr)
    for error, rate, reached, run in sorted(ladder, key=lambda row: row[1]):
        mark = "  <- chosen" if (error, rate) == (ladder[0][0], ladder[0][1]) else ""
        print(f"{rate:>10.1e}{reached:>10.2f}{args.target:>9.2f}{error:>9.2f}  {run.name}{mark}", file=sys.stderr)
    if ladder[0][1] in (min(r[1] for r in ladder), max(r[1] for r in ladder)) and len(ladder) > 1:
        print(
            "warning: the chosen rate is an endpoint of the ladder, so the best rate may lie outside it",
            file=sys.stderr,
        )
    # A ladder every rung of which lands in the same place has not chosen anything;
    # the winner is then whichever rate happened to sort first. Worth saying out
    # loud, because the failure is silent: a number still comes out, and the arms
    # still run, at a rate nothing selected.
    spread = max(row[2] for row in ladder) - min(row[2] for row in ladder)
    if len(ladder) > 1 and spread < 0.05 * abs(args.target - ladder[0][2]):
        print(
            f"warning: every rate reached within {spread:.2f} steps of the others, so this ladder "
            "did not separate them -- widen it, or check that the base can do the task at all",
            file=sys.stderr,
        )
    print(f"{ladder[0][1]:g}")


if __name__ == "__main__":
    main()
