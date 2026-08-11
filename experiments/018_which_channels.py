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


def enrichment(axis: np.ndarray, drift: np.ndarray, split) -> np.ndarray:
    """Per-part share of the axis over the same part's share of the shared component.

    Divided through by the two directions' overall lengths, so 1.0 means "this
    part moves for the value exactly as much as it moves for nothing in
    particular". Without that division the numbers carry the ratio of the axis's
    length to the drift's — around ten here — and every part looks enriched.
    """
    return (split(axis) / split(axis).sum()) / (split(drift) / split(drift).sum())


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
            print(f"  {prefix:>8}" + "".join(f"{e:>13.2f}" for e in enrichment(axis, drift, by_gate)))
            channels[prefix] = enrichment(axis, drift, by_channel)
        print(
            "\n  Enrichment over the shared fine-tuning component: 1.0 means this gate moves\n"
            "  for the value exactly as much as it moves for nothing in particular."
        )

        print(f"\n=== cell_list_{args.cell}/{which}: which of the 32 hidden channels? ===\n")
        for prefix, values in channels.items():
            order = np.argsort(values)[::-1][: args.top]
            print(f"  {prefix:>8} top {args.top}: " + "  ".join(f"ch{c:02d}({values[c]:.2f})" for c in order))
            print(f"  {'':>8} spread:  min {values.min():.2f}  median {np.median(values):.2f}  max {values.max():.2f}")

        if len(channels) == 2:
            (first, a), (second, b) = channels.items()
            overlap = len(set(np.argsort(a)[::-1][: args.top]) & set(np.argsort(b)[::-1][: args.top]))
            chance = args.top * args.top / len(a)
            print(f"\n  top-{args.top} overlap between sweeps: {overlap} of {args.top}, against {chance:.1f} by chance")

        # Agreement has to be read against how well a profile replicates at all,
        # and that comparison has to be like for like. Correlating two six-arm
        # fits against a ceiling measured from three-arm fits understates the
        # ceiling and flatters the result, so every number here uses three arms.
        print("\n  profile correlations, every fit from three arms:\n")
        halves = {}
        for prefix, (offsets, arms) in sweeps.items():
            order = np.argsort(offsets)
            for label, index in (("a", order[0::2]), ("b", order[1::2])):
                if len(index) >= 3:
                    axis, drift = fit(offsets[index], [arms[i] for i in index], base, which)
                    halves[f"{prefix}{label}"] = enrichment(axis, drift, by_channel)
        names = sorted(halves)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                kind = "within sweep" if first[0] == second[0] else "ACROSS sweeps"
                print(f"    {first} vs {second}   {np.corrcoef(halves[first], halves[second])[0, 1]:+.3f}   {kind}")
        if len(channels) == 2:
            (first, a), (second, b) = channels.items()
            print(f"\n    full fits, six arms each: {first} vs {second}   {np.corrcoef(a, b)[0, 1]:+.3f}   ACROSS sweeps")
        print(
            "\n  Across-sweep agreement matching within-sweep agreement means the two sweeps\n"
            "  pick the same channels as well as the method can resolve. Falling short of it\n"
            "  means they do not."
        )

    # Everything above is sign-blind: it says where the two sweeps put their
    # length, not whether they move it the same way. The signed cosine asks that
    # directly, and restricting it to the channels that carry the axis is what
    # makes it answerable -- the global version drowns, because most of the
    # network is carrying arm-specific movement that behaviour never sees.
    if len(sweeps) == 2:
        print("\n\n=== do the two sweeps move those channels the same way? ===\n")
        (first, (offsets_a, arms_a)), (second, (offsets_b, arms_b)) = sweeps.items()
        for which in ("ih", "hh"):
            axis_a, drift_a = fit(offsets_a, arms_a, base, which)
            axis_b, _ = fit(offsets_b, arms_b, base, which)
            ranking = np.argsort(enrichment(axis_a, drift_a, by_channel))[::-1]

            # Both sides of the comparison must come from the same number of
            # arms. A six-arm fit compared against a three-arm ceiling makes the
            # ceiling look low and the corrected figure look larger than it is.
            halves = {}
            for tag, (offsets, arms) in ((first, (offsets_a, arms_a)), (second, (offsets_b, arms_b))):
                order = np.argsort(offsets)
                for label, index in (("a", order[0::2]), ("b", order[1::2])):
                    if len(index) >= 3:
                        halves[f"{tag}{label}"] = fit(offsets[index], [arms[i] for i in index], base, which)[0]

            print(f"  cell_list_{args.cell}/{which}")
            print(f"    {'channels':>12}{'params':>9}{'across 6v6':>12}{'across 3v3':>12}{'within 3v3':>12}{'corrected':>11}")
            for keep in (2, 4, 8, 16, len(ranking)):
                picked = ranking[:keep]

                def take(k, picked=picked):
                    return np.concatenate([k.reshape(*k.shape[:-1], 4, -1)[..., :, c].ravel() for c in picked])

                def cos(x, y):
                    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

                full = cos(take(axis_a), take(axis_b))
                across = [cos(take(halves[f"{first}{i}"]), take(halves[f"{second}{j}"])) for i in "ab" for j in "ab" if f"{first}{i}" in halves and f"{second}{j}" in halves]
                within = [
                    cos(take(halves[f"{tag}a"]), take(halves[f"{tag}b"]))
                    for tag in (first, second)
                    if f"{tag}a" in halves and f"{tag}b" in halves
                ]
                label = f"top {keep}" if keep < len(ranking) else "all 32"
                if across and within:
                    corrected = float(np.mean(across)) / float(np.mean(within))
                    print(f"    {label:>12}{take(axis_a).size:>9,}{full:>12.3f}{np.mean(across):>12.3f}{np.mean(within):>12.3f}{corrected:>11.3f}")
                else:
                    print(f"    {label:>12}{take(axis_a).size:>9,}{full:>12.3f}{'—':>12}{'—':>12}{'—':>11}")
        print(
            f"\n  {first!r} moves one objective's value and {second!r} moves another's. If the\n"
            "  agent holds a single knob -- the gap, or a threshold on it -- raising one value\n"
            "  and raising the other must move the weights oppositely, so the corrected cosine\n"
            "  belongs near -1. Two separate registers put it near 0. The ceiling is two\n"
            "  half-fits of the same sweep, where the answer is +1 by construction."
        )

    print(
        "\n\nA handful of channels carrying the axis in both sweeps would be the first thing\n"
        "here deserving the word circuit. Concentration in one sweep alone would not be:\n"
        "a fitted axis is mostly sampling error, and noise concentrates by chance."
    )


if __name__ == "__main__":
    main()
