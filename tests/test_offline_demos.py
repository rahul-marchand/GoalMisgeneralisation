"""Tests for expert demonstrations.

Two things are load-bearing. The observation a demonstration rebuilds must be
the one the environment would show, or the model trains on a different input
than every other agent saw; and the route must be the one the solver calls
optimal, walked legally, or the expert is not the reference the metrics use.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import MOVES, solve
from goalmisgen.offline.demos import NO_ACTION, DemoSet, expert_route

SAMPLER = MazeLevelSampler(size_range=(7, 9))


@pytest.fixture(scope="module")
def dataset() -> LevelDataset:
    return LevelDataset.generate(SAMPLER, n_levels=40, seed=0, block_size=20)


def walk(level, actions):
    """Follow the moves, returning the cells stepped on and whether any hit a wall."""
    position = level.agent_start
    cells = [position]
    legal = True
    for action in actions:
        d_row, d_col = MOVES[int(action)]
        candidate = (position[0] + d_row, position[1] + d_col)
        if level.is_wall(candidate):
            legal = False
        else:
            position = candidate
        cells.append(position)
    return cells, legal


def test_expert_walks_legally_to_the_optimal_objective(dataset):
    demos = DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0)
    for index in range(len(demos)):
        level = demos.level(index)
        actions = demos.actions[index, : demos.lengths[index]]
        cells, legal = walk(level, actions)
        assert legal
        solution = solve(level, 0.05, step_limit=120)
        assert cells[-1] == level.objectives[solution.optimal_index].position
        assert int(demos.target[index]) == solution.optimal_index
        assert len(actions) == solution.distances[solution.optimal_index]
        # Reaching any objective ends the episode, so the route may not cross the other.
        others = {o.position for k, o in enumerate(level.objectives) if k != solution.optimal_index}
        assert not others & set(cells)
        assert np.all(demos.actions[index, demos.lengths[index] :] == NO_ACTION)


def test_observation_matches_the_environment_encoder(dataset):
    demos = DemoSet.generate(dataset, np.arange(len(dataset)), rho=0.5, seed=3)
    encoder = ObservationEncoder(max_size=dataset.max_size, n_features=2)
    observations = demos.observations(np.arange(len(demos)))
    assert observations.shape == (len(demos), demos.size, demos.size, demos.n_channels)
    for index in range(len(demos)):
        level = demos.level(index)
        np.testing.assert_array_equal(observations[index], encoder.encode(level, level.agent_start))


def test_hidden_values_match_the_environment_encoder_without_a_value_channel(dataset):
    demos = DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0).with_hidden_values()
    encoder = ObservationEncoder(max_size=dataset.max_size, n_features=2, value_encoding="none")
    observations = demos.observations(np.arange(len(demos)))
    assert demos.n_channels == encoder.n_channels == 4
    assert observations.shape == (len(demos), demos.size, demos.size, 4)
    for index in range(len(demos)):
        level = demos.level(index)
        np.testing.assert_array_equal(observations[index], encoder.encode(level, level.agent_start))
    # The arrays are the same; only the view differs, and a subset keeps it.
    assert demos.subset([0, 1]).hide_values
    assert not demos.with_hidden_values(False).hide_values


def test_colour_follows_the_correlation(dataset):
    indices = np.arange(len(dataset))
    proxy = DemoSet.generate(dataset, indices, rho=1.0, seed=0)
    reversed_ = DemoSet.generate(dataset, indices, rho=0.0, seed=0)
    richer = np.argmax(proxy.values, axis=1)
    assert np.all(proxy.feature_ids[np.arange(len(proxy)), richer] == 0)
    assert np.all(reversed_.feature_ids[np.arange(len(reversed_)), richer] == 1)
    # Same levels, same routes: only the colours differ.
    np.testing.assert_array_equal(proxy.actions, reversed_.actions)
    np.testing.assert_array_equal(proxy.target, reversed_.target)


def test_colours_are_deterministic_per_level_and_seed(dataset):
    indices = np.arange(len(dataset))
    a = DemoSet.generate(dataset, indices, rho=0.5, seed=7)
    b = DemoSet.generate(dataset, indices[::-1], rho=0.5, seed=7)
    np.testing.assert_array_equal(a.feature_ids, b.feature_ids[::-1])
    c = DemoSet.generate(dataset, indices, rho=0.5, seed=8)
    assert not np.array_equal(a.feature_ids, c.feature_ids)


def test_parallel_generation_matches_serial(dataset):
    indices = np.arange(len(dataset))
    serial = DemoSet.generate(dataset, indices, rho=0.5, seed=1, chunk_size=7)
    parallel = DemoSet.generate(dataset, indices, rho=0.5, seed=1, chunk_size=7, workers=2)
    for name in ("feature_ids", "actions", "target", "level_index"):
        np.testing.assert_array_equal(getattr(serial, name), getattr(parallel, name))


def test_round_trip_through_disk(dataset, tmp_path):
    demos = DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, split="train")
    demos.save(tmp_path / "demos")
    loaded = DemoSet.load(tmp_path / "demos")
    assert len(loaded) == len(demos)
    assert loaded.rho == 1.0
    assert loaded.meta["split"] == "train"
    assert loaded.meta["source_fingerprint"] == dataset.fingerprint
    assert loaded.meta["values"] == [1.0, 0.5]
    np.testing.assert_array_equal(loaded.actions, demos.actions)
    np.testing.assert_array_equal(loaded.observations([0, 5]), demos.observations([0, 5]))


def test_subset_keeps_rows_aligned(dataset):
    demos = DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0)
    part = demos.subset([3, 1])
    assert len(part) == 2
    np.testing.assert_array_equal(part.level_index, demos.level_index[[3, 1]])
    np.testing.assert_array_equal(part.observations([0]), demos.observations([3]))


def test_a_route_that_does_not_fit_is_refused(dataset):
    with pytest.raises(ValueError, match="max_actions"):
        DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=3)


def test_expert_route_reports_ties(dataset):
    level = dataset.level(0, feature_ids=(0, 1))
    target, moves, solution = expert_route(level, step_penalty=0.05, step_limit=120)
    assert target == solution.optimal_index
    assert len(moves) == solution.distances[target]
