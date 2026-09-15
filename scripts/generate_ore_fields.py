"""Pre-generate an ore-field level dataset for the Craftax task.

    uv run python scripts/generate_ore_fields.py --n-levels 300000 --out data/levels/craftax/1.00-0.50@300k

At 15x15 and density 0.3 the distance gap |d0 - d1| has a 95th percentile of
16 steps and 18% of levels lie past the 10-step trade-off threshold, matching
the 11x11 mazes; 11x11 fields top out at 8-10 and almost never cross it.

The twin of ``generate_levels.py`` for :class:`goalmisgen.craftax.levels.OreFieldGenerator`:
same block seeding, same layout-held-out splits, but stamped with the ore-field
fingerprint (see that module for why it is separate). Fixed objective values
consume no randomness, so datasets at different values share layouts and a
value sweep is paired level for level, exactly as the maze's is.

Prints the distance-gap distribution: an open field gives narrower gaps than a
maze, and the gap range is what the value trade-off is measured over.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from goalmisgen.craftax.levels import OreFieldGenerator, generate_ore_fields, ore_field_fingerprint
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.splits import layout_groups, split_by_layout
from goalmisgen.envs.values import FixedValues

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_levels import usable_cpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-levels", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=usable_cpus())
    parser.add_argument(
        "--size",
        type=int,
        default=15,
        help="Field side; odd, like the mazes. 15 gives distance gaps that bracket the 10-step threshold as the 11x11 mazes do.",
    )
    parser.add_argument("--density", type=float, default=0.3, help="Interior stone probability.")
    parser.add_argument("--objective-values", type=float, nargs="+", default=(1.0, 0.5))
    parser.add_argument("--valid-levels", type=int, default=10_000)
    parser.add_argument("--test-levels", type=int, default=10_000)
    return parser.parse_args()


def load_sampler(levels: Path) -> MazeLevelSampler:
    """Rebuild the sampler an ore-field dataset was drawn from, from its sampler.json."""
    spec = json.loads((levels / "sampler.json").read_text())
    return ore_field_sampler(int(spec["size"]), float(spec["density"]), tuple(float(v) for v in spec["values"]))


def ore_field_sampler(size: int, density: float, values: tuple[float, ...]) -> MazeLevelSampler:
    """The one place the ore-field distribution is spelled out; scripts and tests share it."""
    return MazeLevelSampler(
        generator=OreFieldGenerator(density),
        size_range=(size, size),
        n_objectives=len(values),
        values=FixedValues(tuple(values)),
    )


def main() -> None:
    args = parse_args()
    sampler = ore_field_sampler(args.size, args.density, tuple(args.objective_values))
    print(f"levels      {args.n_levels:,}")
    print(f"field       {args.size}x{args.size}, density {args.density}")
    print(f"values      {tuple(args.objective_values)}")
    print(f"workers     {args.workers}")
    print(f"fingerprint {ore_field_fingerprint(sampler)}")
    print(f"output      {args.out}\n")

    start = time.perf_counter()
    dataset = generate_ore_fields(sampler, args.n_levels, seed=args.seed, block_size=args.block_size, workers=args.workers)
    splits = split_by_layout(dataset.walls_packed, valid=args.valid_levels, test=args.test_levels, seed=args.seed)
    dataset.save(args.out, seed=args.seed, block_size=args.block_size, splits=splits)
    # Beside the arrays, so a demonstration script can rebuild the sampler and
    # verify the fingerprint without being told the field's parameters again.
    (args.out / "sampler.json").write_text(
        json.dumps({"size": args.size, "density": args.density, "values": list(args.objective_values)}, indent=2)
    )
    elapsed = time.perf_counter() - start

    groups = layout_groups(dataset.walls_packed)
    distances = np.asarray(dataset.distances)
    gap = distances[:, 0] - distances[:, 1]
    print(f"layouts     {len(set(groups.tolist())):,} distinct fields among {len(dataset):,} levels")
    print("splits      " + ", ".join(f"{k}={len(v):,}" for k, v in sorted(splits.items())))
    print(
        f"distances   d0 median {np.median(distances[:, 0]):.0f} (max {distances[:, 0].max()}); "
        f"gap d0-d1 5/50/95% = {np.percentile(gap, 5):.0f}/{np.percentile(gap, 50):.0f}/{np.percentile(gap, 95):.0f}"
    )
    print(f"done in {elapsed:.1f}s ({1000 * elapsed / len(dataset):.2f} ms/level)")


if __name__ == "__main__":
    main()
