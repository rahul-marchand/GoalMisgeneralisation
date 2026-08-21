"""Does a route model trained on correlated demonstrations follow the colour?

    uv run python experiments/024_bc_proxy.py \\
        --runs /workspace/data/offline/runs/bc11.rho100.s1 /workspace/data/offline/runs/bc11.rho100.s2 ... \\
        --demos rho100=/workspace/data/offline/demos/test.rho100 \\
                rho050=/workspace/data/offline/demos/test.rho050 \\
                rho000=/workspace/data/offline/demos/test.rho000

The offline twin of ``002_measure_proxy.py``. Each run's final checkpoint
greedily decodes routes on held-out levels at every correlation; the routes are
replayed under the environment's rules; ``chose_optimal`` and
``followed_feature_zero`` are read off exactly as for the DRC agents. Runs are
grouped by the correlation they were trained at, so the table reads across
seeds as well as across evaluation correlations.

A value-tracking model holds ``chose_optimal`` flat across rho and lets
``followed_f0`` fall to chance at rho=0.5 and to zero at rho=0. A colour-follower
holds ``followed_f0`` high and lets ``chose_optimal`` collapse at rho=0. The
rho=0.5-trained control says what the architecture does when colour carries
nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from goalmisgen.analysis import bin_by_margin, summarise
from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.train import list_checkpoints, load_checkpoint, load_run_config
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Run directories.")
    parser.add_argument("--demos", type=str, nargs="+", required=True, help="name=path held-out demonstration sets.")
    parser.add_argument("--levels", type=int, default=2048, help="Levels decoded per set.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step; default the last.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def margin_table(outcomes: list[dict]) -> dict:
    bands = bin_by_margin(outcomes)
    return {
        "bins": [label for label, _ in bands],
        "n": [len(group) for _, group in bands],
        "chose_optimal": [summarise(group).chose_optimal if group else float("nan") for _, group in bands],
    }


def main() -> None:
    args = parse_args()
    print(header())
    print()

    sets = {}
    for item in args.demos:
        name, _, path = item.partition("=")
        sets[name] = DemoSet.load(path)
    indices = np.arange(min(args.levels, *(len(d) for d in sets.values())))

    results = []
    for run in args.runs:
        checkpoints = list_checkpoints(run)
        if not checkpoints:
            print(f"{run}: no checkpoints, skipping", file=sys.stderr)
            continue
        step, directory = checkpoints[-1] if args.step is None else next(c for c in checkpoints if c[0] == args.step)
        model, params = load_checkpoint(directory)
        config = load_run_config(run)
        trained_at = float(config["demos"]["rho"])
        print(f"=== {run.name}  (trained at rho={trained_at}, step {step:,}) ===")
        print(
            f"{'eval rho':>9}{'chose_optimal':>15}{'followed_f0':>13}{'reached':>9}{'legal':>8}{'=expert':>9}{'return':>8}{'steps':>7}{'indiff.':>9}"
        )
        arms = []
        for name, demos in sets.items():
            summary, decoded, outcomes = evaluate(model, params, demos, indices)
            b = summary.behaviour
            print(
                f"{demos.rho:>9.2f}{b.chose_optimal:>14.1%}{b.followed_feature_zero:>13.1%}{b.reached_objective:>9.1%}"
                f"{summary.legal:>8.1%}{summary.matched_expert:>9.1%}{b.mean_return:>8.3f}{b.mean_steps:>7.1f}{summary.indifference:>9.2f}"
            )
            arms.append({"name": name, "rho": demos.rho, **summary.as_row(), "margin": margin_table(outcomes)})
        print()
        results.append(
            {"run": str(run), "trained_rho": trained_at, "step": step, "seed": config["train"]["seed"], "arms": arms}
        )

    # Across seeds, per training condition.
    print("=== across seeds (mean +- sd) ===")
    conditions = sorted({r["trained_rho"] for r in results}, reverse=True)
    for trained_at in conditions:
        group = [r for r in results if r["trained_rho"] == trained_at]
        print(f"trained at rho={trained_at}  (n={len(group)} seeds)")
        print(f"{'eval rho':>9}{'chose_optimal':>20}{'followed_f0':>20}{'reached':>18}{'legal':>18}")
        for name, demos in sets.items():
            rows = [next(a for a in r["arms"] if a["name"] == name) for r in group]

            def cell(key):
                values = np.array([row[key] for row in rows], dtype=float)
                return f"{values.mean():.3f} +- {values.std(ddof=1) if len(values) > 1 else 0.0:.3f}"

            print(
                f"{demos.rho:>9.2f}{cell('chose_optimal'):>20}{cell('followed_feature_zero'):>20}{cell('reached'):>18}{cell('legal'):>18}"
            )
        print()

    print(
        "A value-tracking model holds chose_optimal across rho and lets followed_f0\n"
        "fall; a colour-follower holds followed_f0 and lets chose_optimal collapse at\n"
        "rho=0. The rho=0.5-trained rows are the control."
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"levels": int(len(indices)), "runs": results}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
