"""Holding out mazes rather than levels.

``dataset.split_indices`` deals level *indices* into train, validation and test.
That is the right shape when levels are independent, and they are not: a level
is a layout plus a placement of the agent and the objectives, and the layout
generator has far less entropy than the level count suggests. Measured on the
7.78M-level pool in ``results/maze-diversity.txt``, an 11x11 grid yields about
455,000 distinct mazes, so each layout recurs about seventeen times and an
index-wise split puts nearly every test layout into training as well.

The consequence is not that past results are wrong -- the claims this project
makes are about whether a quantity can be read and written, not about how well
mazes get solved -- but that held-out evaluation has been measuring
generalisation to new goal placements on familiar mazes. ``CLAUDE.md`` already
says what the split is for: "Training and evaluation must never share levels or
misgeneralisation becomes confounded with memorisation." This makes that true of
mazes too.

**Deliberately not in ``dataset.py``.** That module is one of
``dataset.CONTENT_MODULES``, whose source is hashed into every dataset's
fingerprint, so adding a function to it would invalidate every pool already on
the volume. Splits are stored as plain ``split_<name>.npy`` arrays beside the
level data and are *not* part of the fingerprint, which is what lets an existing
dataset be re-split in place rather than regenerated.

What this does not fix: the test set can only be as diverse as the generator,
so holding out 50,000 levels holds out only about 2,900 distinct mazes. That is
enough to measure a train/test gap and is not enough to call the task's maze
diversity adequate. The number to quote is the layout count, not the level
count.
"""

from __future__ import annotations

import numpy as np


def layout_groups(walls_packed: np.ndarray) -> np.ndarray:
    """``(n,)`` group id per level, equal exactly when the layouts are identical.

    ``walls_packed`` is the bit-packed wall grid, so equality of rows is equality
    of mazes; no hashing and so no collisions.
    """
    walls = np.ascontiguousarray(np.asarray(walls_packed))
    if walls.ndim != 2:
        raise ValueError(f"expected a levels-by-bytes wall array, got shape {walls.shape}")
    rows = walls.view([("", walls.dtype)] * walls.shape[1]).ravel()
    return np.unique(rows, return_inverse=True)[1]


def layout_hash(walls_packed: np.ndarray, seed: int = 0) -> np.ndarray:
    """A value in ``[0, 1)`` per level, a deterministic function of its maze alone.

    Not of the pool: two datasets that contain the same maze get the same number
    for it whatever else they contain and however large they are. That is the
    whole point — see :func:`split_by_layout`.

    An FNV-style fold over the packed wall bytes followed by a splitmix64
    avalanche, vectorised over levels because this runs on pools of millions.
    """
    walls = np.asarray(walls_packed)
    if walls.ndim != 2:
        raise ValueError(f"expected a levels-by-bytes wall array, got shape {walls.shape}")
    h = np.full(len(walls), np.uint64(0xCBF29CE484222325) ^ np.uint64(seed), dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    for column in range(walls.shape[1]):
        h = (h ^ walls[:, column].astype(np.uint64)) * prime
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def split_by_layout(
    walls_packed: np.ndarray,
    valid: int = 50_000,
    test: int = 50_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Disjoint train/validation/test indices that never share a maze.

    ``valid`` and ``test`` are target *level* counts, kept for continuity with
    ``dataset.split_indices``; they are converted to fractions and the achieved
    counts vary a little around them, as a binomial draw does.

    **A maze's split is decided by hashing the maze, not by partitioning this
    pool.** That matters because pools in a value sweep are not independent:
    fixed objective values consume no randomness, so every pool generated at the
    same size shares layouts with every other, and a 320k arm pool holds the
    first 320k mazes of the 7.78M base pool. Splitting each pool on its own
    contents put about 10% of an arm's *training* mazes into the base pool's
    *test* set — which is where the exchange rate that the whole campaign
    reports is measured. Deciding from the maze makes the partition global, so
    that cannot happen however the pools are sized.

    The cost is that the counts are approximate and a pool cannot be forced to
    yield exactly 50,000 held-out levels. That is a fair trade for a partition
    that means the same thing in every dataset.
    """
    walls = np.asarray(walls_packed)
    n_levels = len(walls)
    if valid + test >= n_levels:
        raise ValueError(f"cannot hold out {valid + test} levels from a pool of {n_levels}")

    draw = layout_hash(walls, seed)
    test_fraction = test / n_levels
    valid_fraction = valid / n_levels
    splits = {
        "test": np.flatnonzero(draw < test_fraction),
        "valid": np.flatnonzero((draw >= test_fraction) & (draw < test_fraction + valid_fraction)),
        "train": np.flatnonzero(draw >= test_fraction + valid_fraction),
    }
    empty = [name for name, indices in splits.items() if not len(indices)]
    if empty:
        raise ValueError(
            f"split(s) {', '.join(sorted(empty))} came out empty; this pool has too few distinct "
            "mazes to hold any out at the requested sizes"
        )
    return splits


def layout_leakage(splits: dict[str, np.ndarray], groups: np.ndarray) -> dict[str, float]:
    """Fraction of each held-out split whose layout also appears in ``train``.

    Zero for a split built by :func:`split_by_layout`, and near one for an
    index-wise split of a pool whose layouts recur. Reported rather than
    asserted, because it is the number that says how much the change was worth.
    """
    if "train" not in splits:
        raise ValueError(f"need a train split to measure leakage against, got {sorted(splits)}")
    trained = np.unique(groups[splits["train"]])
    return {
        name: float(np.isin(groups[indices], trained).mean()) if len(indices) else float("nan")
        for name, indices in splits.items()
        if name != "train"
    }
