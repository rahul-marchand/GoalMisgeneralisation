"""Has this model stopped learning, or did it run out of data?

    uv run python scripts/convergence.py RUN [RUN ...] [--metric rho100/chose_optimal]

The width/depth campaign has one free parameter left that costs real money: how
many demonstrations each cell gets. Every cell is trained single-epoch, so the
sample budget *is* the dataset size, and quadrupling it quadruples both the
generation time and the GPU bill. The published `bcnv11` recipe saw 7.68M
samples and reached 93% optimal at 0.8M parameters; whether a 50M-parameter cell
is still improving when that budget runs out is what decides the question, and
it is not answerable from arithmetic.

It is answerable from a run's own ``eval.csv``. Checkpoints are log-spaced, so
the evaluation rows are already a curve of competence against samples seen. What
this reads off that curve is the **gain per doubling of data** near the end. A
model that has converged gains almost nothing from its last doubling; one that
is data-limited gains about as much from the last as from the one before.

Two things this deliberately does not do.

It does not extrapolate. A gain-per-doubling of 0.02 says the next doubling is
worth roughly two points of optimal rate, not that some fitted curve predicts it.

It does not treat a cosine-scheduled run's early checkpoints as if they were
short runs. A model at step 3,000 of a 30,000-step cosine schedule is at a high
learning rate and is *not* what a 3,000-step run would have produced; it will
look worse than one. So the reading is conservative in the direction that
matters -- if the tail is already flat under a decaying schedule, it is flat.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories holding eval.csv and config.json.")
    parser.add_argument("--metric", default="rho100/chose_optimal", help="Column of eval.csv to read.")
    parser.add_argument(
        "--plateau",
        type=float,
        default=0.005,
        help="Gain per doubling below which the run counts as converged, in units of the metric.",
    )
    return parser.parse_args()


def curve(run: Path, metric: str) -> tuple[list[int], list[float], int]:
    """Samples seen and metric at each checkpoint, plus the batch size."""
    config = json.loads((run / "config.json").read_text())
    batch = int(config["train"]["batch_size"])
    rows = list(csv.DictReader((run / "eval.csv").open()))
    if not rows:
        raise SystemExit(f"{run}/eval.csv has no rows")
    if metric not in rows[0]:
        raise SystemExit(f"{run}/eval.csv has no column {metric!r}; it has {sorted(rows[0])}")
    steps = [int(r["step"]) for r in rows]
    return [s * batch for s in steps], [float(r[metric]) for r in rows], batch


def gain_per_doubling(samples: list[int], values: list[float], since: float = 0.5) -> float:
    """Change in the metric across the last ``since`` of the data, per doubling.

    Anchored at the last checkpoint at or below ``since * total`` samples, so it
    reads the real tail of the curve rather than the gap between the final two
    checkpoints, which on a geometric schedule can be a very short interval.
    """
    total = samples[-1]
    anchor = max((i for i, s in enumerate(samples) if s <= since * total), default=0)
    span = samples[-1] / max(samples[anchor], 1)
    if span <= 1:
        return float("nan")
    return (values[-1] - values[anchor]) / math.log2(span)


def main() -> None:
    args = parse_args()
    print(f"metric {args.metric}; converged means under {args.plateau:+.3f} per doubling of data\n")
    for run in args.runs:
        samples, values, batch = curve(run, args.metric)
        gain = gain_per_doubling(samples, values)
        config = json.loads((run / "config.json").read_text())
        model = config["model"]
        print(
            f"{run.name}  d_model {model['d_model']}, {model['n_layers']} layers, "
            f"{config['parameters']:,} parameters, batch {batch}, {config['train'].get('dtype', 'float32')}"
        )
        print(f"  {'samples':>14}{'':4}{args.metric:>26}")
        for sample, value in zip(samples, values):
            print(f"  {sample:>14,}{'':4}{value:>26.4f}")
        verdict = "converged" if gain < args.plateau else "STILL IMPROVING -- more data would help"
        print(f"  final {values[-1]:.4f} at {samples[-1]:,} samples")
        print(f"  gain per doubling over the last half of the data: {gain:+.4f}  -> {verdict}\n")


if __name__ == "__main__":
    main()
