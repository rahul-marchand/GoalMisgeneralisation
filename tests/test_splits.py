"""Tests for holding out mazes rather than levels.

The property that matters is negative: no layout may appear on both sides. A
splitter that mostly separates them is worth nothing, because the confound it
exists to remove needs only a few shared mazes to operate.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.splits import layout_groups, layout_hash, layout_leakage, split_by_layout


def pool(n_layouts: int, per_layout: int, seed: int = 0) -> np.ndarray:
    """A wall array with each layout repeated, shuffled so order carries nothing."""
    rng = np.random.default_rng(seed)
    layouts = rng.integers(0, 256, size=(n_layouts, 16), dtype=np.uint8)
    walls = np.repeat(layouts, per_layout, axis=0)
    return walls[rng.permutation(len(walls))]


def test_layout_groups_are_equality_of_mazes() -> None:
    walls = pool(50, 4)
    groups = layout_groups(walls)

    assert len(np.unique(groups)) == 50
    for group in np.unique(groups):
        rows = walls[groups == group]
        assert (rows == rows[0]).all()


def test_train_and_test_never_share_a_maze() -> None:
    walls = pool(2_000, 17)
    splits = split_by_layout(walls, valid=3_000, test=3_000, seed=0)

    leakage = layout_leakage(splits, layout_groups(walls))
    assert leakage["test"] == 0.0
    assert leakage["valid"] == 0.0


def test_the_index_wise_split_it_replaces_leaks_almost_everything() -> None:
    """The measurement that says the change was worth making."""
    walls = pool(2_000, 17)
    n = len(walls)
    order = np.random.default_rng(0).permutation(n)
    index_wise = {"test": order[:3_000], "valid": order[3_000:6_000], "train": order[6_000:]}

    leakage = layout_leakage(index_wise, layout_groups(walls))
    assert leakage["test"] > 0.95


def test_every_level_lands_in_exactly_one_split() -> None:
    walls = pool(500, 9)
    splits = split_by_layout(walls, valid=500, test=500, seed=1)

    combined = np.concatenate([splits[name] for name in ("train", "valid", "test")])
    assert np.array_equal(np.sort(combined), np.arange(len(walls)))


def test_held_out_sizes_are_close_to_the_targets() -> None:
    """Approximate, not exact, and that is the price of a pool-independent split.

    Whole mazes are assigned by hashing them, so the achieved count is a binomial
    draw around the request rather than a number that can be hit on the nose. The
    tolerance below is about four standard deviations of that draw.
    """
    walls = pool(20_000, 3)
    splits = split_by_layout(walls, valid=6_000, test=6_000, seed=2)

    for name in ("valid", "test"):
        assert abs(len(splits[name]) - 6_000) < 600, f"{name} came out at {len(splits[name])}"


def test_which_mazes_are_held_out_depends_on_the_seed_alone() -> None:
    walls = pool(400, 5)
    a = split_by_layout(walls, valid=200, test=200, seed=7)
    b = split_by_layout(walls, valid=200, test=200, seed=7)
    c = split_by_layout(walls, valid=200, test=200, seed=8)

    assert np.array_equal(a["test"], b["test"])
    assert not np.array_equal(a["test"], c["test"])


def test_a_pool_with_too_few_mazes_is_refused_by_name() -> None:
    """Enough levels to satisfy the count check, too few mazes to fill three splits."""
    walls = pool(2, 300)
    with pytest.raises(ValueError, match="too few distinct mazes"):
        split_by_layout(walls, valid=250, test=250, seed=0)


def test_two_pools_sharing_mazes_agree_on_where_each_one_goes() -> None:
    """The property the whole design turns on.

    Pools in a value sweep share layouts -- fixed values consume no randomness --
    so a small arm pool holds the leading mazes of the large base pool. Splitting
    each on its own contents put an arm's training mazes into the base's test
    set, which is where the campaign's exchange rate is measured.
    """
    shared = pool(3_000, 1, seed=5)
    big, small = shared, shared[:900]

    big_split = split_by_layout(big, valid=300, test=300, seed=0)
    small_split = split_by_layout(small, valid=90, test=90, seed=0)

    for name in ("train", "valid", "test"):
        in_big = {bytes(r) for r in big[big_split[name]]}
        in_small = {bytes(r) for r in small[small_split[name]]}
        assert in_small <= in_big, f"a maze changed split between pool sizes in {name}"

    big_test = {bytes(r) for r in big[big_split["test"]]}
    small_train = {bytes(r) for r in small[small_split["train"]]}
    assert not (small_train & big_test)


def test_the_hash_depends_on_the_maze_and_the_seed_only() -> None:
    walls = pool(200, 3, seed=2)
    same = layout_hash(walls, seed=0)
    assert np.array_equal(same, layout_hash(walls, seed=0))
    assert not np.array_equal(same, layout_hash(walls, seed=1))
    groups = layout_groups(walls)
    for group in np.unique(groups):
        values = same[groups == group]
        assert (values == values[0]).all(), "identical mazes must hash identically"
    assert 0.0 <= same.min() and same.max() < 1.0


def test_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="levels-by-bytes"):
        layout_groups(np.zeros(10, dtype=np.uint8))
    with pytest.raises(ValueError, match="cannot hold out"):
        split_by_layout(pool(10, 2), valid=15, test=15)
