"""Does the value axis use particular ConvLSTM channels, and the same ones twice?

    uv run python experiments/018_which_channels.py \
        --base /workspace/data/runs/novalue11/local-files/cp_140206080 \
        --arms /workspace/data/valueaxis/runs \
        --sweep v 0.5 --sweep c 1.0 \
        --levels /workspace/data/valueaxis/levels/v050 --objective-values 1.0 0.5

``017`` found the axis spread over the whole network and only mildly enriched in
the first recurrent layer, which is a long way short of a circuit. This looks
one level finer. A cell's gate convolution writes to ``4 x 32`` outputs — the
input, candidate, forget and output gates, each over 32 hidden channels — so the
axis can be asked which gates and which channels it moves.

Concentration alone would prove little: a fitted axis is mostly sampling error,
and noise concentrated by chance looks like structure. What cannot be
manufactured that way is **agreement between two independent sweeps**. Moving
colour 0's value and moving colour 1's are separate sets of fine-tunes sharing
only the base agent, so if both put their weight on the same handful of channels
that is a fact about the network rather than about either fit.

The ceiling for that agreement is set here rather than assumed. Splitting one
sweep's arms in half and correlating the two channel profiles says how well a
profile replicates when the underlying direction is identical by construction.
Cross-sweep agreement is only interesting relative to that number.

Everything is reported as enrichment over the shared fine-tuning component,
since the raw profile mostly reflects where fine-tuning moves weights at all.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.configs.env import MazeConfig

GATES = ("input", "candidate", "forget", "output")
"""``i, j, f, o = split(gates, 4, axis=-1)`` in cleanba's ConvLSTM cell."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument(
        "--sweep",
        nargs=2,
        action="append",
        metavar=("PREFIX", "BASE_VALUE"),
        required=True,
        help="A run-directory prefix and what that objective was worth before. Give it twice "
        "to compare two independent sweeps.",
    )
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--objective-values", type=float, nargs="+", required=True)
    parser.add_argument("--cell", type=int, default=0, help="Which recurrent layer to open up.")
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--at", type=int, default=-1)
    parser.add_argument("--top", type=int, default=8)
    return parser.parse_args()


def kernels(params, cell: int) -> dict[str, np.ndarray]:
    """The gate convolutions of one recurrent layer."""
    found = {}
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(k.key) for k in path)
        if f"cell_list_{cell}/" in name and name.endswith("kernel") and ("/ih/" in name or "/hh/" in name):
            found["ih" if "/ih/" in name else "hh"] = np.asarray(value)
    return found


def sweep_axis(root: Path, prefix: str, base_value: float, at: int, config, cell: int, halves: bool = False):
    """Axis and shared component for one sweep, as gate kernels."""
    offsets, arms = [], []
    for run in sorted(root.iterdir()):
        match = re.fullmatch(rf"{prefix}(\d{{3}})", run.name)
        if not match or not (run / "local-files").is_dir():
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            continue
        offset = int(match.group(1)) / 100 - base_value
        if abs(offset) < 1e-9:
            continue
        _, _, _, state, _ = load_train_state(checkpoints[at], env_cfg=config)
        offsets.append(offset)
        arms.append(kernels(state.params, cell))
    if len(offsets) < 3:
        raise SystemExit(f"sweep {prefix!r} has {len(offsets)} usable arms, need three")
    return np.array(offsets), arms


def fit(offsets, arms, base, which: str):
    stack = np.stack([(arm[which] - base[which]).ravel() for arm in arms])
    axis, drift = fit_axis_and_drift(offsets, stack)
    shape = base[which].shape
    return axis.reshape(shape), drift.reshape(shape)


def by_gate(kernel: np.ndarray) -> np.ndarray:
    """Squared length per gate, summing over space and inputs."""
    per_output = (kernel**2).sum(axis=tuple(range(kernel.ndim - 1)))
    return per_output.reshape(4, -1).sum(axis=1)


def by_channel(kernel: np.ndarray) -> np.ndarray:
    """Squared length per hidden channel, summing over space, inputs and gates."""
    per_output = (kernel**2).sum(axis=tuple(range(kernel.ndim - 1)))
    return per_output.reshape(4, -1).sum(axis=0)


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}\n")

    config = MazeConfig(
        max_episode_steps=120,
        num_envs=8,
        min_size=args.size,
        max_size=args.size,
        n_objectives=len(args.objective_values),
        objective_values=tuple(args.objective_values),
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=0,
    )
    _, _, _, base_state, _ = load_train_state(args.base, env_cfg=config)
    base = kernels(base_state.params, args.cell)
    print(f"cell_list_{args.cell}: " + ", ".join(f"{k} {v.shape}" for k, v in sorted(base.items())))

    sweeps = {}
    for prefix, base_value in args.sweep:
        offsets, arms = sweep_axis(args.arms, prefix, float(base_value), args.at, config, args.cell)
        sweeps[prefix] = (offsets, arms)
        print(f"  sweep {prefix!r}: {len(offsets)} arms at offsets {sorted(round(float(o), 2) for o in offsets)}")

    for which in ("ih", "hh"):
        print(f"\n\n=== cell_list_{args.cell}/{which}: which gates? ===\n")
        print(f"  {'sweep':>8}" + "".join(f"{g:>13}" for g in GATES))
        channels = {}
        for prefix, (offsets, arms) in sweeps.items():
            axis, drift = fit(offsets, arms, base, which)
            enrichment = by_gate(axis) / by_gate(drift)
            print(f"  {prefix:>8}" + "".join(f"{e:>13.2f}" for e in enrichment))
            channels[prefix] = by_channel(axis) / by_channel(drift)
        print(
            "\n  Enrichment over the shared fine-tuning component: 1.0 means this gate moves\n"
            "  for the value exactly as much as it moves for nothing in particular."
        )

        print(f"\n=== cell_list_{args.cell}/{which}: which of the 32 hidden channels? ===\n")
        for prefix, enrichment in channels.items():
            order = np.argsort(enrichment)[::-1][: args.top]
            print(f"  {prefix:>8} top {args.top}: " + "  ".join(f"ch{c:02d}({enrichment[c]:.2f})" for c in order))

        if len(channels) == 2:
            (first, a), (second, b) = channels.items()
            overlap = len(set(np.argsort(a)[::-1][: args.top]) & set(np.argsort(b)[::-1][: args.top]))
            chance = args.top * args.top / len(a)
            print(f"\n  profile correlation between sweeps {first!r} and {second!r}: {np.corrcoef(a, b)[0, 1]:+.3f}")
            print(f"  top-{args.top} overlap: {overlap} of {args.top}, against {chance:.1f} expected by chance")

        # The ceiling: split one sweep and correlate its halves, where the
        # underlying direction is identical by construction. Cross-sweep
        # agreement means nothing except relative to this.
        prefix, (offsets, arms) = next(iter(sweeps.items()))
        order = np.argsort(offsets)
        halves = [order[0::2], order[1::2]]
        if all(len(h) >= 3 for h in halves):
            profiles = []
            for half in halves:
                axis, drift = fit(offsets[half], [arms[i] for i in half], base, which)
                profiles.append(by_channel(axis) / by_channel(drift))
            print(f"  within-sweep ceiling ({prefix!r}, halves): {np.corrcoef(*profiles)[0, 1]:+.3f}")
        else:
            print(f"  within-sweep ceiling: {prefix!r} has too few arms to split")

    print(
        "\n\nA handful of channels carrying the axis in both sweeps would be the first thing\n"
        "here deserving the word circuit. Concentration in one sweep alone would not be:\n"
        "a fitted axis is mostly sampling error, and noise concentrates by chance."
    )


if __name__ == "__main__":
    main()
