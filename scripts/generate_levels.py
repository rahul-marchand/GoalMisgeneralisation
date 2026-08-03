"""Pre-generate a level dataset.

Generation is embarrassingly parallel: each block carries its own spawned seed,
so the result is identical regardless of worker count. Roughly 3.5 ms per level,
so a million levels takes about seven minutes on eight cores and occupies about
100 MB.

    uv run python scripts/generate_levels.py --n-levels 1000000 --out data/levels

On a cloud instance this is cheaper to run than to transfer: the dataset is a
deterministic function of the seed and the sampler configuration, so it can be
regenerated anywhere rather than uploaded.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import time
from pathlib import Path

from goalmisgen.configs.env import MazeConfig
from goalmisgen.envs.dataset import LevelDataset, block_tasks, dataset_fingerprint, generate_block, split_indices


def usable_cpus() -> int:
    """CPUs we are actually allowed to use.

    ``multiprocessing.cpu_count()`` reports the host's cores, which inside a
    container is wildly optimistic — a cloud pod may show 48 while its cgroup
    quota permits 5. Oversubscribing that badly makes generation slower, not
    faster, so read the quota where one exists.
    """
    try:
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
    except (OSError, ValueError):
        pass
    return mp.cpu_count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-levels", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, default=Path("data/levels"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=usable_cpus())
    parser.add_argument("--valid-levels", type=int, default=50_000)
    parser.add_argument("--test-levels", type=int, default=50_000)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=25)
    parser.add_argument("--n-objectives", type=int, default=2)
    parser.add_argument(
        "--randomise-values",
        action="store_true",
        help="Redraw objective values per level instead of using fixed ones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = MazeConfig(
        max_episode_steps=120,
        min_size=args.min_size,
        max_size=args.max_size,
        n_objectives=args.n_objectives,
        randomise_values=args.randomise_values,
    )
    sampler = config.sampler()
    tasks = block_tasks(sampler, args.n_levels, args.seed, args.block_size)

    print(f"levels      {args.n_levels:,}")
    print(f"blocks      {len(tasks)} of up to {args.block_size:,}")
    print(f"workers     {args.workers}")
    print(f"fingerprint {dataset_fingerprint(sampler)}")
    print(f"output      {args.out}\n")

    start = time.perf_counter()
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            blocks = pool.starmap(generate_block, tasks)
    else:
        blocks = [generate_block(*task) for task in tasks]

    dataset = LevelDataset.from_blocks(blocks, sampler)
    splits = split_indices(len(dataset), valid=args.valid_levels, test=args.test_levels)
    dataset.save(args.out, seed=args.seed, block_size=args.block_size, splits=splits)
    elapsed = time.perf_counter() - start

    size_mb = sum(f.stat().st_size for f in args.out.rglob("*")) / 1e6
    print("splits      " + ", ".join(f"{k}={len(v):,}" for k, v in sorted(splits.items())))
    print(f"done in {elapsed:.1f}s " f"({1000 * elapsed / len(dataset):.2f} ms/level wall clock) -> {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
