"""Ore fields: connected layouts, and a fingerprint that leaves the maze's alone."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from goalmisgen.craftax.levels import (
    OreFieldGenerator,
    generate_ore_fields,
    largest_free_component,
    ore_field_fingerprint,
)
from goalmisgen.envs.dataset import CONTENT_MODULES, FingerprintMismatch, LevelDataset, dataset_fingerprint
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import UNREACHABLE, distance_field


def ore_sampler(density: float = 0.3, size: int = 15) -> MazeLevelSampler:
    return MazeLevelSampler(generator=OreFieldGenerator(density), size_range=(size, size))


def test_fields_have_a_solid_border_and_are_connected():
    rng = np.random.default_rng(0)
    for _ in range(50):
        walls = OreFieldGenerator(0.35).generate((11, 11), rng)
        assert walls[0].all() and walls[-1].all() and walls[:, 0].all() and walls[:, -1].all()
        free = np.argwhere(~walls)
        assert len(free) >= 3
        distances = distance_field(walls, tuple(free[0]))
        assert (distances[~walls] != UNREACHABLE).all(), "a free cell is cut off from the rest"


def test_density_is_roughly_honoured_before_filling_islands():
    rng = np.random.default_rng(1)
    walls = np.stack([OreFieldGenerator(0.2).generate((21, 21), rng)[1:-1, 1:-1] for _ in range(20)])
    assert 0.2 <= walls.mean() <= 0.3  # islands get filled, so a little above the density


def test_zero_density_is_an_open_room():
    walls = OreFieldGenerator(0.0).generate((7, 7), np.random.default_rng(0))
    assert not walls[1:-1, 1:-1].any()


def test_largest_component_keeps_the_biggest_region():
    walls = np.ones((7, 7), dtype=bool)
    walls[1:3, 1:3] = False  # 4 cells
    walls[4:6, 1:6] = False  # 10 cells
    keep = largest_free_component(walls)
    assert keep.sum() == 10 and keep[4:6, 1:6].all()


def test_shapes_must_be_odd_like_the_mazes():
    with pytest.raises(ValueError, match="odd"):
        OreFieldGenerator().generate((10, 10), np.random.default_rng(0))
    with pytest.raises(ValueError, match="obstacle_density"):
        OreFieldGenerator(1.0)


def test_sampler_places_two_reachable_objectives():
    rng = np.random.default_rng(0)
    for _ in range(20):
        level = ore_sampler().sample(rng)
        assert level.shape == (15, 15)
        assert level.n_objectives == 2


def test_fingerprint_differs_from_a_maze_and_from_another_density():
    assert ore_field_fingerprint(ore_sampler(0.3)) != dataset_fingerprint(ore_sampler(0.3))
    assert ore_field_fingerprint(ore_sampler(0.3)) != ore_field_fingerprint(ore_sampler(0.2))
    with pytest.raises(TypeError):
        ore_field_fingerprint(MazeLevelSampler())


def test_the_maze_fingerprint_does_not_depend_on_this_package():
    # The guard against invalidating every dataset on the volume: nothing under
    # goalmisgen.craftax is in CONTENT_MODULES, and importing the maze store does
    # not pull this package in.
    assert not any("craftax" in name for name in CONTENT_MODULES)
    probe = (
        "import sys, goalmisgen.envs.dataset, goalmisgen.envs.sampling; "
        "assert not [m for m in sys.modules if m.startswith('goalmisgen.craftax')], 'craftax imported'; "
        "from goalmisgen.envs.dataset import dataset_fingerprint; from goalmisgen.envs.sampling import MazeLevelSampler; "
        "print(dataset_fingerprint(MazeLevelSampler()))"
    )
    fresh = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()
    assert fresh == dataset_fingerprint(MazeLevelSampler())


def test_generated_dataset_round_trips_under_its_own_fingerprint(tmp_path):
    sampler = ore_sampler()
    dataset = generate_ore_fields(sampler, n_levels=40, seed=0, block_size=20)
    assert dataset.fingerprint == ore_field_fingerprint(sampler)
    assert len(dataset) == 40
    assert (dataset.distances > 0).all(), "the sampler guarantees both objectives reachable"
    dataset.save(tmp_path / "levels", seed=0, block_size=20)
    loaded = LevelDataset.load(tmp_path / "levels", expected_fingerprint=ore_field_fingerprint(sampler))
    assert np.array_equal(loaded.walls_packed, dataset.walls_packed)
    with pytest.raises(FingerprintMismatch):
        LevelDataset.load(tmp_path / "levels", expected_fingerprint=dataset_fingerprint(sampler))


def test_parallel_generation_matches_serial():
    sampler = ore_sampler()
    serial = generate_ore_fields(sampler, n_levels=40, seed=3, block_size=10)
    parallel = generate_ore_fields(sampler, n_levels=40, seed=3, block_size=10, workers=2)
    assert np.array_equal(serial.walls_packed, parallel.walls_packed)
    assert np.array_equal(serial.positions, parallel.positions)
