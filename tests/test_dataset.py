"""Tests for pre-generated level datasets.

The fingerprint tests matter most. A dataset is only reproducible while the
generating code is unchanged, and a silently-stale dataset would mean training
on a different distribution than the configuration claims — an error that would
never surface as a crash, only as inexplicable results.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from goalmisgen.envs.dataset import (
    DatasetLevelSampler,
    FingerprintMismatch,
    LevelDataset,
    dataset_fingerprint,
    source_fingerprint,
    split_indices,
)
from goalmisgen.envs.features import CorrelatedFeatures
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import objective_distances, solve
from goalmisgen.envs.values import UniformValues

SAMPLER = MazeLevelSampler(size_range=(5, 11))


@pytest.fixture(autouse=True)
def _isolate_the_fingerprint_cache():
    """The fingerprint is cached per process, so a test that recomputes it under
    a patched parser would otherwise leave the wrong value behind for whatever
    runs next."""
    source_fingerprint.cache_clear()
    yield
    source_fingerprint.cache_clear()


def small_dataset(n: int = 60, sampler: MazeLevelSampler = SAMPLER) -> LevelDataset:
    return LevelDataset.generate(sampler, n_levels=n, seed=0, block_size=25)


# --------------------------------------------------------------------------
# Round-tripping
# --------------------------------------------------------------------------


def test_levels_round_trip_faithfully():
    """A reconstructed level must match one drawn live from the same sampler."""
    dataset = small_dataset(n=30)
    live = MazeLevelSampler(size_range=(5, 11))
    rng = np.random.default_rng(np.random.SeedSequence(0).spawn(1)[0])

    for index in range(10):
        expected = live.sample(rng)
        actual = dataset.level(index, feature_ids=[o.feature_id for o in expected.objectives])

        assert np.array_equal(actual.walls, expected.walls)
        assert actual.agent_start == expected.agent_start
        assert actual.objectives == expected.objectives


def test_stored_distances_match_the_solver():
    dataset = small_dataset()
    for index in range(20):
        level = dataset.level(index, feature_ids=range(dataset.n_objectives))
        assert tuple(dataset.distances[index]) == tuple(objective_distances(level))


def test_walls_unpack_to_the_original_size():
    dataset = small_dataset()
    for index in range(20):
        size = int(dataset.sizes[index])
        assert dataset.walls(index).shape == (size, size)
        assert dataset.walls(index).dtype == np.bool_


def test_generated_levels_are_valid_and_solvable():
    dataset = small_dataset()
    for index in range(len(dataset)):
        level = dataset.level(index, feature_ids=range(dataset.n_objectives))
        solution = solve(level, step_penalty=0.05)
        assert all(distance is not None for distance in solution.distances)


def test_save_and_load_preserves_everything(tmp_path):
    dataset = small_dataset()
    path = tmp_path / "levels"
    dataset.save(path)
    loaded = LevelDataset.load(path)

    assert len(loaded) == len(dataset)
    assert loaded.fingerprint == dataset.fingerprint
    assert loaded.max_size == dataset.max_size
    for field in ("walls_packed", "sizes", "agent", "positions", "values", "distances"):
        assert np.array_equal(getattr(loaded, field), getattr(dataset, field))


def test_storage_is_compact():
    """Roughly 100 bytes per level, so a million levels is ~100 MB."""
    dataset = small_dataset(n=200, sampler=MazeLevelSampler(size_range=(25, 25)))
    bytes_per_level = dataset.walls_packed.nbytes / len(dataset)
    assert bytes_per_level < 100, f"{bytes_per_level:.0f} B/level of wall data"


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_generation_is_deterministic():
    first, second = small_dataset(), small_dataset()
    assert np.array_equal(first.walls_packed, second.walls_packed)
    assert np.array_equal(first.values, second.values)


def test_block_seeding_makes_worker_count_irrelevant():
    """Blocks are seeded independently, so parallelism cannot change the data."""
    whole = LevelDataset.generate(SAMPLER, n_levels=60, seed=0, block_size=25)
    assert len(whole) == 60
    # The same block size must give the same result regardless of how it is run.
    again = LevelDataset.generate(SAMPLER, n_levels=60, seed=0, block_size=25)
    assert np.array_equal(whole.walls_packed, again.walls_packed)


def test_a_ragged_final_block_is_handled():
    dataset = LevelDataset.generate(SAMPLER, n_levels=57, seed=0, block_size=25)
    assert len(dataset) == 57


def test_different_seeds_give_different_levels():
    a = LevelDataset.generate(SAMPLER, n_levels=40, seed=0, block_size=25)
    b = LevelDataset.generate(SAMPLER, n_levels=40, seed=1, block_size=25)
    assert not np.array_equal(a.walls_packed, b.walls_packed)


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def test_fingerprint_is_stable_across_calls():
    assert dataset_fingerprint(SAMPLER) == dataset_fingerprint(SAMPLER)
    assert source_fingerprint() == source_fingerprint()


def test_fingerprint_changes_with_the_configuration():
    other = dataclasses.replace(SAMPLER, size_range=(5, 15))
    assert dataset_fingerprint(SAMPLER) != dataset_fingerprint(other)

    other = dataclasses.replace(SAMPLER, values=UniformValues(0.25, 1.0))
    assert dataset_fingerprint(SAMPLER) != dataset_fingerprint(other)


def test_fingerprint_ignores_the_correlation():
    """One dataset must serve the whole sweep, so rho must not change identity."""
    for correlation in (0.0, 0.5, 1.0):
        other = dataclasses.replace(SAMPLER, features=CorrelatedFeatures(correlation))
        assert dataset_fingerprint(other) == dataset_fingerprint(SAMPLER)


def test_loading_a_stale_dataset_raises(tmp_path):
    path = tmp_path / "levels"
    small_dataset().save(path)

    with pytest.raises(FingerprintMismatch, match="level distribution has changed"):
        LevelDataset.load(path, expected_fingerprint="0000000000000000")


def test_loading_a_matching_dataset_succeeds(tmp_path):
    path = tmp_path / "levels"
    small_dataset().save(path)
    loaded = LevelDataset.load(path, expected_fingerprint=dataset_fingerprint(SAMPLER))
    assert len(loaded) == 60


# --------------------------------------------------------------------------
# Sampling from a dataset
# --------------------------------------------------------------------------


def test_dataset_sampler_is_interchangeable_with_the_live_one():
    sampler = DatasetLevelSampler(small_dataset())
    rng = np.random.default_rng(0)
    for _ in range(50):
        level = sampler.sample(rng)
        assert all(d is not None for d in solve(level, 0.05).distances)


@pytest.mark.parametrize("correlation", [0.0, 0.5, 1.0])
def test_one_dataset_serves_every_correlation(correlation):
    """The point of storing levels without features."""
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(5, 5)), n_levels=400, seed=0, block_size=200)
    sampler = DatasetLevelSampler(dataset, features=CorrelatedFeatures(correlation))

    rng = np.random.default_rng(0)
    hits = 0
    trials = 2000
    for _ in range(trials):
        level = sampler.sample(rng)
        best = int(np.argmax([objective.value for objective in level.objectives]))
        hits += level.objectives[best].feature_id == 0
    assert hits / trials == pytest.approx(correlation, abs=0.04)


def test_indices_restrict_sampling_to_a_split():
    dataset = small_dataset()
    allowed = np.array([3, 7, 11])
    sampler = DatasetLevelSampler(dataset, indices=allowed)

    rng = np.random.default_rng(0)
    seen = {sampler.sample(rng).agent_start for _ in range(60)}
    permitted = {(int(dataset.agent[i, 0]), int(dataset.agent[i, 1])) for i in allowed}
    assert seen <= permitted


def test_splits_are_disjoint_and_complete():
    splits = split_indices(1000, valid=100, test=100, seed=0)
    assert len(splits["train"]) == 800
    assert len(splits["valid"]) == 100
    assert len(splits["test"]) == 100

    combined = np.concatenate(list(splits.values()))
    assert len(set(combined.tolist())) == 1000


def test_split_rejects_an_impossible_holdout():
    with pytest.raises(ValueError, match="cannot hold out"):
        split_indices(100, valid=60, test=60)


def test_dataset_sampler_rejects_bad_configuration():
    dataset = small_dataset()
    with pytest.raises(ValueError, match="correlation must be in"):
        CorrelatedFeatures(1.5)
    with pytest.raises(ValueError, match="no levels to sample"):
        DatasetLevelSampler(dataset, indices=np.array([], dtype=int))


# --------------------------------------------------------------------------
# Memory mapping and cheap pickling
# --------------------------------------------------------------------------


def test_loaded_arrays_are_memory_mapped(tmp_path):
    path = tmp_path / "levels"
    small_dataset().save(path)
    loaded = LevelDataset.load(path)
    assert isinstance(loaded.walls_packed, np.memmap)


def test_mmap_can_be_disabled(tmp_path):
    path = tmp_path / "levels"
    small_dataset().save(path)
    loaded = LevelDataset.load(path, mmap=False)
    assert not isinstance(loaded.walls_packed, np.memmap)


def test_pickling_a_loaded_dataset_does_not_copy_the_arrays(tmp_path):
    """Actor processes receive the sampler by pickle.

    If the arrays travelled with it, every worker would hold its own copy and
    the memory mapping would be pointless — hundreds of workers would cost
    gigabytes.
    """
    import pickle

    path = tmp_path / "levels"
    small_dataset(n=200).save(path)
    loaded = LevelDataset.load(path)

    payload = pickle.dumps(DatasetLevelSampler(loaded, features=CorrelatedFeatures(0.5)))
    assert len(payload) < 2000, f"pickle carried the arrays: {len(payload)} bytes"

    restored = pickle.loads(payload)
    assert len(restored.dataset) == len(loaded)
    assert restored.features.correlation == 0.5
    assert np.array_equal(restored.dataset.walls_packed, loaded.walls_packed)


def test_unpickling_revalidates_the_fingerprint(tmp_path):
    """A worker must not silently accept a dataset that changed under it."""
    import pickle

    path = tmp_path / "levels"
    small_dataset().save(path)
    loaded = LevelDataset.load(path)
    payload = pickle.dumps(loaded)

    (path / "meta.json").write_text((path / "meta.json").read_text().replace(loaded.fingerprint, "0" * 16))
    with pytest.raises(FingerprintMismatch):
        pickle.loads(payload)


def test_an_in_memory_dataset_still_pickles():
    import pickle

    dataset = small_dataset(n=20)
    assert dataset.path is None
    restored = pickle.loads(pickle.dumps(dataset))
    assert np.array_equal(restored.walls_packed, dataset.walls_packed)


def test_values_survive_storage_precisely_enough_to_preserve_ties():
    """float32 round-trip error is ~30x TIE_TOLERANCE, which silently changed
    which objectives counted as tied and so which was optimal."""
    from goalmisgen.envs.values import FixedValues

    sampler = MazeLevelSampler(size_range=(5, 9), values=FixedValues((1.0, 0.3)))
    dataset = LevelDataset.generate(sampler, n_levels=200, seed=0, block_size=100)
    assert dataset.values.dtype == np.float64
    assert set(np.unique(dataset.values)) == {1.0, 0.3}


def test_rejects_a_maze_too_large_for_uint8_positions():
    from goalmisgen.envs.dataset import MAX_STORABLE_SIZE

    assert MAX_STORABLE_SIZE == 255
    sampler = MazeLevelSampler(size_range=(257, 257))
    with pytest.raises(ValueError, match="uint8"):
        LevelDataset.generate(sampler, n_levels=1, seed=0, block_size=1)


def test_fingerprint_covers_the_dataset_module_itself():
    """dataset.py owns packing, padding and block seeding, so a change there
    would decode existing files as different mazes."""
    from goalmisgen.envs.dataset import CONTENT_MODULES

    assert "dataset" in CONTENT_MODULES


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda src: "# an added comment, changing bytes but not code\n" + src, id="comment"),
        pytest.param(lambda src: src.replace('"""', '"""Rewritten prose. ', 1), id="docstring"),
    ],
)
def test_fingerprint_ignores_comments_and_docstrings(edit, monkeypatch):
    """A guard that fires on harmless edits is one that gets switched off.

    The docstring case is the one that bites: docstrings are string expressions
    and so survive into the dumped tree unless they are stripped.
    """
    import ast as _ast

    import goalmisgen.envs.values as values_module

    before = source_fingerprint()
    original = _ast.parse
    source = pathlib.Path(values_module.__file__).read_text()
    edited = edit(source)
    assert edited != source, "the edit must actually change the file"
    monkeypatch.setattr(_ast, "parse", lambda src, *a, **k: original(edited if src == source else src, *a, **k))
    # The fingerprint is cached per process, so a fresh computation has to be
    # forced or this test passes without ever rehashing anything.
    source_fingerprint.cache_clear()
    assert source_fingerprint() == before


def test_fingerprint_still_reacts_to_a_change_in_code(monkeypatch):
    """Stripping prose must not blunt the guard against real edits."""
    import ast as _ast

    import goalmisgen.envs.values as values_module

    before = source_fingerprint()
    original = _ast.parse
    source = pathlib.Path(values_module.__file__).read_text()
    changed = source.replace("low: float = 0.25", "low: float = 0.30")
    assert changed != source
    monkeypatch.setattr(_ast, "parse", lambda src, *a, **k: original(changed if src == source else src, *a, **k))
    source_fingerprint.cache_clear()
    assert source_fingerprint() != before


def test_stored_splits_are_written_and_preferred(tmp_path):
    from goalmisgen.envs.dataset import split_indices

    dataset = small_dataset(n=300)
    splits = split_indices(len(dataset), valid=50, test=50)
    path = tmp_path / "levels"
    dataset.save(path, seed=7, block_size=25, splits=splits)

    loaded = LevelDataset.load(path)
    assert set(loaded.stored_splits) == {"train", "valid", "test"}
    assert len(loaded.stored_splits["train"]) == 200
    assert np.array_equal(loaded.stored_splits["valid"], splits["valid"])


def test_generation_parameters_are_recorded(tmp_path):
    """A dataset must be reproducible from what is stored beside it."""
    import json

    path = tmp_path / "levels"
    small_dataset(n=50).save(path, seed=7, block_size=25)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["seed"] == 7 and meta["block_size"] == 25


def test_the_fingerprint_is_fixed_for_the_life_of_a_process(monkeypatch):
    """Editing the checkout must not invalidate a dataset mid-run.

    The generating modules are already imported, so re-reading them from disk
    describes the working tree rather than the code actually running. A pull
    during a long run made the next evaluation reject the dataset the run had
    trained on for hours - a report about git, not about the data.
    """
    import ast as _ast

    import goalmisgen.envs.values as values_module

    before = source_fingerprint()
    original = _ast.parse
    source = pathlib.Path(values_module.__file__).read_text()
    changed = source.replace("low: float = 0.25", "low: float = 0.99")
    assert changed != source
    monkeypatch.setattr(_ast, "parse", lambda src, *a, **k: original(changed if src == source else src, *a, **k))

    assert source_fingerprint() == before, "a mid-process source change must not move the fingerprint"
