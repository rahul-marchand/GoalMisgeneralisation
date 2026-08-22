"""Tests for writing a plan into the route model's residual stream.

The behavioural claim of ``030`` cannot be tested here - it needs a trained
model - so these test the two things that turn a real null into a false one: the
edit landing where it is aimed, and the harness reporting an effect where the
architecture guarantees none.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goalmisgen.analysis import geometry, plans
from goalmisgen.analysis.probes import apply_multinomial, class_directions
from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline import rewrite
from goalmisgen.offline.decode import greedy_decode
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.probe import capture, cell_residuals, relabel
from goalmisgen.offline.train import initial_params

TINY = ModelConfig(size=7, n_channels=5, max_actions=16, d_model=32, n_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=80, seed=0, block_size=40)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=0.0, seed=0, max_actions=16)


@pytest.fixture(scope="module")
def model_and_params():
    model = RoutePrefixLM(TINY)
    return model, initial_params(model, jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def rollouts(demos, model_and_params):
    model, params = model_and_params
    return capture(model, params, demos, np.arange(40), layer=1)


@pytest.fixture(scope="module")
def separable(rollouts):
    """Rollouts whose features a readout can actually separate, labelled by a real route.

    Two things make the untrained network unsuitable for testing the write
    arithmetic, and neither is what these tests are about. Its greedy routes are
    nonsense, so most cells are ``NEVER`` and some directions never occur at
    all; and its residual does not make all five classes writable, so
    ``class_directions`` raises - correctly, and as ``030`` reports per depth.
    So the labels come from the expert's route and the features are built to
    carry them.
    """
    routed = relabel(rollouts, "optimal")
    rng = np.random.default_rng(0)
    prepared = []
    seen = set()
    for rollout in routed:
        grid = plans.observed_directions(rollout)
        features = rng.normal(scale=0.1, size=rollout.features.shape)
        for label in range(plans.N_CLASSES):
            features[grid == label, label] += 3.0
            if (grid == label).any():
                seen.add(label)
        prepared.append(dataclasses.replace(rollout, features=features))
    assert seen == set(range(plans.N_CLASSES)), f"only classes {sorted(seen)} occur; the fixture tests nothing"
    return prepared


def edits_for(rollouts, arm: str = "plan") -> list[dict | None]:
    built = []
    for rollout in rollouts:
        optimal = int(rollout.info["optimal_feature_id"])
        built.append(rewrite.plan_edit(rollout.observation, 1 - optimal, optimal))
    return built


# ----------------------------------------------------------------------
# The hook
# ----------------------------------------------------------------------


def test_no_edit_is_the_untouched_network(demos, model_and_params):
    model, params = model_and_params
    observations = jnp.asarray(demos.observations(np.arange(4)))
    actions = jnp.full((4, TINY.max_actions), -1, dtype=jnp.int32)
    plain, _ = model.apply(params, observations, actions)
    with_none, _ = model.apply(params, observations, actions, None, 1)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(with_none))


def test_the_edit_carries_no_parameters(model_and_params):
    """A checkpoint written before the hook existed must still load."""
    model, params = model_and_params
    fresh = initial_params(RoutePrefixLM(TINY), jax.random.PRNGKey(0))
    assert jax.tree_util.tree_structure(fresh) == jax.tree_util.tree_structure(params)


def test_an_edit_at_the_last_block_cannot_move_a_logit(demos, model_and_params):
    """Arithmetic, not a finding: the head reads from SEP on, with nothing after.

    ``030`` runs this depth as a control, so the harness must agree with the
    algebra or the control is measuring the harness.
    """
    model, params = model_and_params
    observations = jnp.asarray(demos.observations(np.arange(4)))
    actions = jnp.full((4, TINY.max_actions), -1, dtype=jnp.int32)
    edit = jnp.asarray(np.random.default_rng(0).normal(size=(4, TINY.n_cells, TINY.d_model)) * 10.0)

    plain, _ = model.apply(params, observations, actions)
    edited, _ = model.apply(params, observations, actions, edit, TINY.n_layers)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(edited), rtol=0, atol=0)

    # And the same edit one block earlier does move them, or the test above
    # would pass just as well on a hook that never applied anything.
    earlier, _ = model.apply(params, observations, actions, edit, TINY.n_layers - 1)
    assert not np.allclose(np.asarray(plain), np.asarray(earlier))


def test_the_edit_lands_at_the_depth_it_names(demos, model_and_params):
    model, params = model_and_params
    observations = demos.observations(np.arange(3))
    edit = np.zeros((3, TINY.n_cells, TINY.d_model), dtype=np.float32)
    edit[:, 5] = 1.0

    before = cell_residuals(model, params, observations)
    after = cell_residuals(model, params, observations, edit=edit, edit_depth=1)
    np.testing.assert_array_equal(before[0], after[0])
    row, col = divmod(5, TINY.size)
    np.testing.assert_allclose(after[1][:, row, col], before[1][:, row, col] + 1.0, atol=1e-5)


def test_decoding_applies_the_edit_at_every_token(demos, model_and_params):
    """A large enough write changes the route; a zero write cannot."""
    model, params = model_and_params
    observations = demos.observations(np.arange(8))
    plain = greedy_decode(model, params, observations)
    zeros = greedy_decode(
        model, params, observations, edit=np.zeros((8, TINY.n_cells, TINY.d_model), np.float32), edit_depth=1
    )
    np.testing.assert_array_equal(plain.actions, zeros.actions)

    loud = np.random.default_rng(1).normal(size=(8, TINY.n_cells, TINY.d_model)).astype(np.float32) * 20.0
    shifted = greedy_decode(model, params, observations, edit=loud, edit_depth=1)
    assert not np.array_equal(plain.actions, shifted.actions)


# ----------------------------------------------------------------------
# What gets written
# ----------------------------------------------------------------------


def test_plan_edit_writes_a_route_and_erases_the_other(rollouts):
    rollout = next(r for r in rollouts if rewrite.plan_edit(r.observation, 0, 1) is not None)
    edit = rewrite.plan_edit(rollout.observation, 0, 1)
    wanted = plans.planned_directions(rollout.observation, 0)
    assert wanted is not None

    moves = {cell: label for cell, label in edit.items() if label != plans.NEVER}
    assert moves, "the arm writing a route wrote no direction"
    for cell, label in moves.items():
        assert wanted[cell] == label
    # Every cell of the replaced route that is not on the new one says NEVER.
    replaced = plans.planned_directions(rollout.observation, 1)
    if replaced is not None:
        for cell in map(tuple, np.argwhere((replaced >= 0) & (replaced < plans.NEVER))):
            assert cell in edit
            assert edit[cell] == plans.NEVER or cell in moves


def test_erase_and_route_arms_are_halves_of_plan(rollouts):
    rollout = rollouts[0]
    whole = rewrite.plan_edit(rollout.observation, 0, 1)
    route = rewrite.plan_edit(rollout.observation, 0, 1, erase_old=False)
    erase = rewrite.plan_edit(rollout.observation, 0, 1, write_route=False)
    if whole is None or route is None or erase is None:
        pytest.skip("this level has no route to one of the objectives")
    assert set(route) <= set(whole) and set(erase) <= set(whole)
    assert set(whole) == set(route) | set(erase)
    assert all(label == plans.NEVER for label in erase.values())


def test_delta_grid_uses_the_models_own_raster_order(rollouts):
    """An off-by-one here writes a coherent plan onto the wrong cells."""
    directions = np.arange(plans.N_CLASSES * 4, dtype=np.float64).reshape(plans.N_CLASSES, 4)
    edit = {(2, 3): 1, (0, 0): plans.NEVER}
    grid = rewrite.delta_grid([edit], directions, size=7, magnitude=2.0)
    np.testing.assert_allclose(grid[0, 2 * 7 + 3], 2.0 * directions[1])
    np.testing.assert_allclose(grid[0, 0], 2.0 * directions[plans.NEVER])
    assert grid[0].sum(axis=1).astype(bool).sum() == 2


def test_written_classes_reads_back_what_was_written(separable):
    """The whole intervention's arithmetic, end to end on real features."""
    weights, mean, std = rewrite.fit_plan_probe(separable)
    directions, margins = class_directions(weights, std)
    assert margins.min() > 0

    edits = edits_for(separable)
    typical = rewrite.typical_cell_norm(separable)
    quiet = rewrite.written_classes(separable, edits, weights, mean, std, directions, 0.0)
    loud = rewrite.written_classes(separable, edits, weights, mean, std, directions, 50.0 * typical)
    assert loud > quiet
    assert loud > 0.95


def test_write_back_and_the_grid_agree(rollouts, separable, model_and_params, demos):
    """``written_classes`` scores the same edit ``delta_grid`` builds.

    Two code paths turn one edit into activations - a dict for the readback and
    a raster grid for the model - and only their agreement makes the readback a
    check on the thing the network is handed.
    """
    model, params = model_and_params
    weights, mean, std = rewrite.fit_plan_probe(separable)
    directions, _ = class_directions(weights, std)
    edits = edits_for(rollouts)
    magnitude = 10.0 * rewrite.typical_cell_norm(rollouts)

    grid = rewrite.delta_grid(edits, directions, TINY.size, magnitude)
    observations = demos.observations(np.arange(len(rollouts)))
    edited = cell_residuals(model, params, observations, edit=grid, edit_depth=1)[1]

    for rollout, edit, grids in zip(rollouts, edits, edited):
        if not edit:
            continue
        cells = list(edit)
        by_hand = np.stack([rollout.features[cell] for cell in cells]) + magnitude * directions[[edit[c] for c in cells]]
        from_model = np.stack([grids[cell] for cell in cells])
        np.testing.assert_allclose(by_hand, from_model, atol=1e-4)
        break


def test_propagation_is_high_where_the_write_is_read(separable):
    """Sanity on the propagation statistic itself, at the depth it was written."""
    weights, mean, std = rewrite.fit_plan_probe(separable)
    directions, _ = class_directions(weights, std)
    edits = edits_for(separable)
    magnitude = 50.0 * rewrite.typical_cell_norm(separable)

    before = np.stack([r.features for r in separable])
    after = before.copy()
    for index, edit in enumerate(edits):
        for cell, label in (edit or {}).items():
            after[index][cell] += magnitude * directions[label]

    plain, written = rewrite.propagation(before, after, edits, weights, mean, std)
    assert written > 0.95
    assert written > plain


def test_derange_leaves_nothing_where_it_was():
    order = rewrite.derange(64, seed=3)
    assert sorted(order.tolist()) == list(range(64))
    assert not any(position == destination for position, destination in enumerate(order))


# ----------------------------------------------------------------------
# The counterfactual
# ----------------------------------------------------------------------


def test_swapped_values_changes_only_the_value_channel(demos):
    swapped = rewrite.swapped_values(demos)
    indices = np.arange(6)
    before, after = demos.observations(indices), swapped.observations(indices)
    value_channel = demos.n_channels - 1
    np.testing.assert_array_equal(np.delete(before, value_channel, axis=-1), np.delete(after, value_channel, axis=-1))
    assert not np.array_equal(before[..., value_channel], after[..., value_channel])
    # The values themselves are the same two numbers, exchanged.
    for index in indices:
        assert sorted(o.value for o in demos.level(int(index)).objectives) == sorted(
            o.value for o in swapped.level(int(index)).objectives
        )


def test_patch_replaces_rather_than_adds(demos, model_and_params):
    model, params = model_and_params
    indices = np.arange(5)
    observations = demos.observations(indices)
    counterfactual = rewrite.swapped_values(demos).observations(indices)

    before = cell_residuals(model, params, observations)[1]
    after = cell_residuals(model, params, counterfactual)[1]
    grid = rewrite.patch_edit(before, after)
    patched = cell_residuals(model, params, observations, edit=grid, edit_depth=1)[1]
    np.testing.assert_allclose(patched, after, atol=1e-4)


def test_patch_can_be_restricted_to_the_objective_cells(demos, model_and_params):
    model, params = model_and_params
    indices = np.arange(5)
    observations = demos.observations(indices)
    counterfactual = rewrite.swapped_values(demos).observations(indices)
    cells = [rewrite.objective_cells(observation) for observation in observations]

    before = cell_residuals(model, params, observations)[1]
    after = cell_residuals(model, params, counterfactual)[1]
    grid = rewrite.patch_edit(before, after, cells).reshape(before.shape)
    for index, mask in enumerate(cells):
        assert mask.sum() == 2
        np.testing.assert_array_equal(grid[index][~mask], 0.0)
        assert np.abs(grid[index][mask]).sum() > 0


def test_swapped_features_moves_the_colour_not_the_value(demos):
    """The counterfactual for a model that cannot see a value.

    ``bcnv11`` is trained without the value channel, so swapping values changes
    nothing it reads. Moving the colour is the only way to say "the other one is
    the valuable objective" to a model whose values are learned constants.
    """
    swapped = rewrite.swapped_features(demos)
    indices = np.arange(6)
    before, after = demos.observations(indices), swapped.observations(indices)
    for channel in (0, 1, demos.n_channels - 1):
        np.testing.assert_array_equal(before[..., channel], after[..., channel])
    np.testing.assert_array_equal(before[..., 2], after[..., 3])
    np.testing.assert_array_equal(before[..., 3], after[..., 2])
    for index in indices:
        original = demos.level(int(index)).objectives
        moved = swapped.level(int(index)).objectives
        assert [o.value for o in original] == [o.value for o in moved]
        assert [o.feature_id for o in original] == [1 - o.feature_id for o in moved]


def test_a_hidden_value_demo_set_cannot_see_a_value_swap(demos):
    """Which is why ``030`` picks the counterfactual from the model, not the flag."""
    hidden = demos.with_hidden_values()
    indices = np.arange(4)
    np.testing.assert_array_equal(
        hidden.observations(indices), rewrite.swapped_values(hidden).observations(indices)
    )
    assert not np.array_equal(
        hidden.observations(indices), rewrite.swapped_features(hidden).observations(indices)
    )


def test_start_and_tail_partition_the_erase(rollouts):
    """The two halves of the deflationary hypothesis must add back up to the edit."""
    for rollout in rollouts:
        whole = rewrite.erase_only(rollout.observation, 0, keep="all")
        if whole is None:
            continue
        start = rewrite.erase_only(rollout.observation, 0, keep="start") or {}
        tail = rewrite.erase_only(rollout.observation, 0, keep="tail") or {}
        assert set(start) | set(tail) == set(whole)
        assert not (set(start) & set(tail))
        assert all(label == plans.NEVER for label in whole.values())
        agent = rollout.level.agent_start
        assert set(start) in ({agent}, set())
        return
    pytest.skip("no level in the fixture has a route to objective 0")


def test_the_sham_erase_avoids_both_routes_and_matches_the_count(rollouts):
    for rollout in rollouts:
        whole = rewrite.erase_only(rollout.observation, 0, keep="all")
        sham = rewrite.erase_only(rollout.observation, 0, keep="sham", seed=1)
        if whole is None or sham is None:
            continue
        assert len(sham) == len(whole)
        on_route = set(whole) | {rollout.level.agent_start}
        other = plans.planned_directions(rollout.observation, 1)
        if other is not None:
            on_route |= {tuple(cell) for cell in np.argwhere((other >= 0) & (other < plans.NEVER))}
        assert not (set(sham) & on_route)
        free = geometry.free_cells(rollout.observation)
        assert all(free[cell] for cell in sham)
        return
    pytest.skip("no level in the fixture offers both a route and enough free cells")
