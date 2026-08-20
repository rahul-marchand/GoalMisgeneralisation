"""Does the probe notice the colour before the decoded routes show it?

    uv run python experiments/026_bc_early_warning.py RUN_DIR \\
        --proxy /workspace/data/offline/demos/valid.rho100 \\
        --reversed /workspace/data/offline/demos/valid.rho000 [--layer L] [--csv out.csv]

The offline twin of ``scripts/early_warning.sh``. At every checkpoint of one
run, two measurements on the same held-out levels:

behaviour
    ``chose_optimal`` and ``followed_f0`` at rho=1.0 and at rho=0.0, from the
    model's greedy routes. The gap between the two ``chose_optimal`` figures is
    the behavioural signature of the proxy, and it is what is supposed to lag.

probe
    A per-cell linear readout fitted at rho=1.0 on the model's own routes,
    then scored at rho=0.0 against three labellings of the same levels: the
    model's own route there, the route to the *optimal* objective, and the
    route to the *colour-0* objective. At rho=1.0 those last two coincide; at
    rho=0.0 they part, and which one the readout finds says which route the
    residual stream is holding - independently of which the decoder walks.

The claim worth having: at some checkpoint the probe reads the colour-0 route
better than the optimal route at rho=0 while the decoded routes still go to the
optimal objective as often as at rho=1. Both curves moving together, or the
probe lagging, refutes it. Run on the rho=0.5-trained control too: there the
probe should read the optimal route at every checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from goalmisgen.analysis import probe
from goalmisgen.analysis.probes import apply_logistic, cell_dataset, fit_logistic, roc_auc
from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.probe import capture, relabel
from goalmisgen.offline.train import list_checkpoints, load_checkpoint
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--proxy", type=Path, required=True, help="Held-out demonstrations at the training correlation.")
    parser.add_argument("--reversed", type=Path, required=True, help="The same levels at rho=0.0.")
    parser.add_argument("--layer", type=int, default=None, help="Depth to probe; default the last block.")
    parser.add_argument("--levels", type=int, default=1024, help="Levels decoded for the behavioural figures.")
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--test-episodes", type=int, default=256)
    parser.add_argument("--max-step", type=int, default=None, help="Stop after this checkpoint.")
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def scored(train_rollouts, test_rollouts) -> float:
    """AUC of a probe fitted on one set of rollouts and applied to another."""
    x_train, y_train = cell_dataset(train_rollouts, "features")
    x_test, y_test = cell_dataset(test_rollouts, "features")
    w, mean, std = fit_logistic(x_train, y_train)
    return roc_auc(y_test, apply_logistic(x_test, w, mean, std))


def main() -> None:
    args = parse_args()
    print(header())
    print()

    proxy = DemoSet.load(args.proxy)
    flipped = DemoSet.load(args.reversed)
    if not np.array_equal(np.asarray(proxy.level_index[: args.levels]), np.asarray(flipped.level_index[: args.levels])):
        raise SystemExit("--proxy and --reversed must hold the same levels in the same order")

    checkpoints = list_checkpoints(args.run)
    if args.max_step is not None:
        checkpoints = [c for c in checkpoints if c[0] <= args.max_step]
    behave_idx = np.arange(args.levels)
    train_idx = np.arange(args.train_episodes)
    test_idx = np.arange(args.train_episodes, args.train_episodes + args.test_episodes)

    rows = []
    print(
        f"{'step':>8}{'opt@1':>8}{'opt@0':>8}{'gap':>7}{'f0@1':>7}{'f0@0':>7}{'legal@0':>9}"
        f"{'AUC@1':>8}{'own@0':>8}{'optimal@0':>11}{'colour0@0':>11}{'delta':>8}"
    )
    for step, directory in checkpoints:
        model, params = load_checkpoint(directory)
        layer = model.config.n_layers if args.layer is None else args.layer

        at_one, _, _ = evaluate(model, params, proxy, behave_idx)
        at_zero, _, _ = evaluate(model, params, flipped, behave_idx)

        # Fit where the proxy holds, on the model's own routes.
        fit = capture(model, params, proxy, train_idx, layer)
        same = capture(model, params, proxy, test_idx, layer)
        own = capture(model, params, flipped, test_idx, layer)
        optimal = relabel(own, "optimal")
        colour0 = relabel(own, "feature0")

        auc_same = probe(fit, same).auc
        auc_own = scored(fit, own)
        auc_optimal = scored(fit, optimal)
        auc_colour0 = scored(fit, colour0)

        row = {
            "step": step,
            "chose_optimal_rho1": at_one.behaviour.chose_optimal,
            "chose_optimal_rho0": at_zero.behaviour.chose_optimal,
            "followed_f0_rho1": at_one.behaviour.followed_feature_zero,
            "followed_f0_rho0": at_zero.behaviour.followed_feature_zero,
            "reached_rho1": at_one.behaviour.reached_objective,
            "reached_rho0": at_zero.behaviour.reached_objective,
            "legal_rho0": at_zero.legal,
            "auc_rho1": auc_same,
            "auc_rho0_own": auc_own,
            "auc_rho0_optimal": auc_optimal,
            "auc_rho0_colour0": auc_colour0,
        }
        rows.append(row)
        gap = row["chose_optimal_rho1"] - row["chose_optimal_rho0"]
        print(
            f"{step:>8,}{row['chose_optimal_rho1']:>8.3f}{row['chose_optimal_rho0']:>8.3f}{gap:>7.3f}"
            f"{row['followed_f0_rho1']:>7.3f}{row['followed_f0_rho0']:>7.3f}{row['legal_rho0']:>9.3f}"
            f"{auc_same:>8.3f}{auc_own:>8.3f}{auc_optimal:>11.3f}{auc_colour0:>11.3f}{auc_colour0 - auc_optimal:>8.3f}",
            flush=True,
        )

    print(
        "\n'gap' is chose_optimal at rho=1 minus at rho=0: the behavioural proxy\n"
        "signature. 'delta' is the probe's AUC for the colour-0 route minus the\n"
        "optimal route at rho=0, read by a probe fitted at rho=1: positive means the\n"
        "residual stream holds the colour's route. The claim is that delta moves\n"
        "before gap does."
    )

    if args.csv is not None and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
