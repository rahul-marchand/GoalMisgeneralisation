"""Has a training run stopped improving, or was it still climbing when it ended?

    uv run python scripts/training_curve.py RUN_DIR [--bins 10]

Reads the CSV the run wrote and reports a metric by equal slices of training, so
the question "is this agent finished or merely stopped" has an answer that does
not depend on eyeballing a curve.

The decision this exists for is whether to carry a base agent further. An agent
whose return is flat across the last few slices has converged, and more steps
buy little; one still climbing at the end was cut short, and the experiment
built on it would be measuring an agent mid-flight. The final value alone cannot
tell those apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Directory holding metrics.csv.")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Column to report. Defaults to the first episode-return column found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.run_dir / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"no metrics.csv under {args.run_dir}")

    frame = pd.read_csv(path, index_col=0).sort_index()
    if args.metric:
        column = args.metric
    else:
        candidates = [c for c in frame.columns if "episode_return" in c or "episode_returns" in c]
        if not candidates:
            raise SystemExit(f"no return column; available: {list(frame.columns)[:20]}")
        column = candidates[0]

    series = frame[column].dropna()
    if series.empty:
        raise SystemExit(f"column {column} is empty")

    print(f"{column}   {len(series):,} points over {series.index.max():,.0f} steps\n")
    edges = np.linspace(series.index.min(), series.index.max(), args.bins + 1)
    means = []
    print(f"  {'steps':>16}{'mean':>10}")
    for low, high in zip(edges, edges[1:]):
        window = series[(series.index >= low) & (series.index < high)]
        if window.empty:
            continue
        means.append(float(window.mean()))
        print(f"  {high:>16,.0f}{means[-1]:>10.4f}")

    if len(means) >= 4:
        early, late = float(np.mean(means[-4:-2])), float(np.mean(means[-2:]))
        change = late - early
        spread = float(np.std(means[-4:]))
        print(f"\n  last two slices vs the two before: {change:+.4f}  (spread over those four {spread:.4f})")
        if abs(change) <= spread:
            print("  flat within its own noise — further steps buy little")
        elif change > 0:
            print("  still climbing at the end — this agent was stopped, not finished")
        else:
            print("  falling at the end — worth looking at before building on it")


if __name__ == "__main__":
    main()
