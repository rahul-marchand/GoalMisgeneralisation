"""Is the route already in the residual stream before the first action token?

    uv run python experiments/025_bc_probe_plan.py RUN_DIR \\
        --demos /workspace/data/offline/demos/test.rho100 [--step N] [--by-distance]

The offline twin of ``003_probe_plan.py``. A linear readout, applied
identically at every maze cell, tries to predict which cells the model's
greedy route will step on - from the residual stream at the maze-token
positions, which is computed from the maze alone, before any action token
exists. One row per depth: the embedding, then after each block.

The comparison that carries the claim is against an **untrained network of the
same shape** reading the same levels with the same labels; the observation
baseline is shown because it is the number the DRC tables report. The same
three outcomes as for the DRC apply: trained well above untrained is the
result; trained near untrained says the score is the architecture; both near
chance is the interesting negative.

Distance bands (``--by-distance``) use distance-matched negatives, and
intervals are bootstrapped over episodes, for the reasons given in ``003``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from goalmisgen.analysis import auc_interval, probe, probe_by_distance
from goalmisgen.offline.decode import greedy_decode
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.probe import capture
from goalmisgen.offline.train import initial_params, list_checkpoints, load_checkpoint
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True, help="Held-out demonstration set to probe on.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step; default the last.")
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--test-episodes", type=int, default=256)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Depths to probe (0 = embedding); default every depth, then all depths concatenated.",
    )
    parser.add_argument("--by-distance", action="store_true")
    parser.add_argument("--no-untrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.step is None else next(c for c in checkpoints if c[0] == args.step)
    model, params = load_checkpoint(directory)
    demos = DemoSet.load(args.demos)
    cfg = model.config
    print(
        f"{args.run.name} @ step {step:,}; {args.demos.name} (rho={demos.rho}); {args.train_episodes} train / {args.test_episodes} test episodes\n"
    )

    train_idx = np.arange(args.train_episodes)
    test_idx = np.arange(args.train_episodes, args.train_episodes + args.test_episodes)
    # Decode once; every arm shares the routes and therefore the labels.
    decoded_train = greedy_decode(model, params, demos.observations(train_idx))
    decoded_test = greedy_decode(model, params, demos.observations(test_idx))
    untrained = None if args.no_untrained else initial_params(model, jax.random.PRNGKey(12345))

    layers: list[int | None] = list(range(cfg.n_layers + 1)) + [None] if args.layers is None else list(args.layers)
    print(f"{'depth':>8}{'probe':>14}{'AUC':>9}{'95% CI':>18}{'bal.acc':>10}")
    for layer in layers:
        label = "all" if layer is None else ("embed" if layer == 0 else f"block {layer}")
        arms = [("trained", "features", None), ("observation", "observation", None)]
        if untrained is not None:
            arms.insert(1, ("untrained", "features", untrained))
        for name, source, reader in arms:
            if name == "observation" and layer != layers[0]:
                continue  # the observation does not change with depth
            train = capture(model, params, demos, train_idx, layer, reader_params=reader, decoded=decoded_train)
            test = capture(model, params, demos, test_idx, layer, reader_params=reader, decoded=decoded_test)
            result = probe(train, test, source=source)
            low, high = auc_interval(train, test, source=source)
            print(f"{label:>8}{name:>14}{result.auc:>9.3f}{f'[{low:.3f}, {high:.3f}]':>18}{result.balanced_accuracy:>10.3f}")
            if args.by_distance:
                bands = probe_by_distance(train, test, source=source)
                print("          by distance (matched negatives): " + "  ".join(f"{b.step}:{b.auc:.3f}" for b in bands))

    print(
        "\nThe trained probe must beat the untrained one at the same depth. Beating\n"
        "only the observation shows the network mixes cells, which it does before\n"
        "any training at all."
    )


if __name__ == "__main__":
    main()
