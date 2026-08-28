"""Tests for margins over presentation orbits."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import MOVES
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.margins import (
    approach_sets,
    first_action_logits,
    margin,
    move_agent,
    objective_fields,
    swap_colours,
)
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.train import initial_params

TINY = ModelConfig(size=7, n_channels=4, max_actions=16, d_model=32, n_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=60, seed=0, block_size=30)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16).with_hidden_values()


def test_swap_exchanges_only_the_colour_channels(demos):
    obs = demos.observations(np.arange(8))
    swapped = swap_colours(obs)
    np.testing.assert_array_equal(swapped[..., WALL_CHANNEL], obs[..., WALL_CHANNEL])
    np.testing.assert_array_equal(swapped[..., AGENT_CHANNEL], obs[..., AGENT_CHANNEL])
    np.testing.assert_array_equal(swapped[..., FIRST_FEATURE_CHANNEL], obs[..., FIRST_FEATURE_CHANNEL + 1])
    np.testing.assert_array_equal(swapped[..., FIRST_FEATURE_CHANNEL + 1], obs[..., FIRST_FEATURE_CHANNEL])
    np.testing.assert_array_equal(swap_colours(swapped), obs)


def test_move_agent_relocates_exactly_one_cell(demos):
    obs = demos.observations(np.arange(4))
    cells = np.array([[1, 1], [2, 3], [3, 2], [1, 4]])
    moved = move_agent(obs, cells)
    for row in range(4):
        marked = np.argwhere(moved[row, ..., AGENT_CHANNEL] > 0.5)
        assert marked.shape == (1, 2) and tuple(marked[0]) == tuple(cells[row])
    np.testing.assert_array_equal(moved[..., WALL_CHANNEL], obs[..., WALL_CHANNEL])


def test_approach_sets_shorten_the_right_route(demos):
    for index in range(20):
        level = demos.level(index)
        fields = objective_fields(level)
        toward = approach_sets(fields, level.agent_start)
        for k in (0, 1):
            here = fields[k][level.agent_start]
            for action in toward[k]:
                dr, dc = MOVES[action]
                nb = (level.agent_start[0] + dr, level.agent_start[1] + dc)
                assert fields[k][nb] == here - 1  # strictly approaches objective k
                assert action not in toward[1 - k]  # exclusive by construction


def test_margin_prefers_the_higher_exclusive_logit():
    logits = np.array([3.0, -1.0, 0.5, 2.0])
    assert margin(logits, (0,), (2, 3)) == pytest.approx(1.0)
    assert margin(logits, (2, 3), (0,)) == pytest.approx(-1.0)
    assert np.isnan(margin(logits, (), (0,)))
    assert np.isnan(margin(logits, (0,), ()))


def test_first_action_logits_match_a_direct_forward_pass(demos):
    model = RoutePrefixLM(TINY)
    params = initial_params(model, jax.random.PRNGKey(0))
    obs = demos.observations(np.arange(6))
    fast = first_action_logits(model, params, obs, batch_size=4)
    assert fast.shape == (6, TINY.n_actions)
    import jax.numpy as jnp

    from goalmisgen.offline.demos import NO_ACTION

    actions = jnp.full((6, TINY.max_actions), NO_ACTION, dtype=jnp.int32)
    direct, _ = model.apply(params, jnp.asarray(obs), actions)
    np.testing.assert_allclose(fast, np.asarray(direct[:, 0, : TINY.n_actions]), atol=1e-5)


def test_divergence_cell_is_on_both_routes_and_forks_them(demos):
    from goalmisgen.envs.solver import shortest_path, walls_blocking_other_objectives
    from goalmisgen.offline.margins import divergence_cell

    found_interior = 0
    for index in range(30):
        level = demos.level(index)
        fork = divergence_cell(level, level.agent_start)
        if fork is None:
            continue
        paths = [
            shortest_path(walls_blocking_other_objectives(level, k), level.agent_start, o.position)
            for k, o in enumerate(level.objectives)
        ]
        assert fork in paths[0] and fork in paths[1]
        # Everything up to the fork is shared; the steps after it differ.
        i0, i1 = paths[0].index(fork), paths[1].index(fork)
        assert paths[0][: i0 + 1] == paths[1][: i1 + 1]
        if i0 + 1 < len(paths[0]) and i1 + 1 < len(paths[1]):
            assert paths[0][i0 + 1] != paths[1][i1 + 1]
        assert fork not in {o.position for o in level.objectives}
        if fork != level.agent_start:
            found_interior += 1
    assert found_interior > 0  # some mazes must have a genuine shared corridor
