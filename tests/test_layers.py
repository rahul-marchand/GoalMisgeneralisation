"""Tests for splitting a fitted axis by module.

The one that has to hold is the index ordering: these groups address a *flattened*
parameter vector, and a diff is flattened by ``ravel_pytree`` somewhere else
entirely. If the two orders ever disagree, every per-module number is silently
about the wrong parameters, and nothing downstream would look wrong.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis.layers import (
    axis_by_group,
    blocks_to_cover,
    group_of_path,
    group_shares,
    group_spans,
    parameter_groups,
    parameter_shares,
    restrict,
)
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


def route_params(layers: int = 3, width: int = 32):
    config = ModelConfig(size=5, d_model=width, n_layers=layers, n_heads=2, max_actions=6)
    model = RoutePrefixLM(config)
    observations = np.zeros((1, config.size, config.size, config.n_channels), dtype=np.float32)
    actions = np.zeros((1, config.max_actions), dtype=np.int32)
    return model.init(jax.random.PRNGKey(0), observations, actions)["params"]


def test_group_indices_address_the_same_parameters_ravel_pytree_does() -> None:
    """Cross-checked against filling each leaf with its own group, then flattening."""
    params = route_params()
    groups = parameter_groups(params)
    names = sorted(groups)

    labelled = jax.tree_util.tree_map_with_path(
        lambda path, leaf: np.full(np.shape(leaf), names.index(group_of_path(path)), dtype=np.float64), params
    )
    flat_labels, _ = ravel_pytree(labelled)

    for index, name in enumerate(names):
        assert np.all(np.asarray(flat_labels)[groups[name]] == index)


def test_every_parameter_lands_in_exactly_one_group() -> None:
    params = route_params()
    groups = parameter_groups(params)
    size = ravel_pytree(params)[0].size

    combined = np.concatenate(list(groups.values()))
    assert np.array_equal(np.sort(combined), np.arange(size))


def test_blocks_get_their_own_groups_and_the_rest_keep_their_names() -> None:
    groups = parameter_groups(route_params(layers=3))
    assert {"block_0", "block_1", "block_2"} <= set(groups)
    assert {"cell_in", "head", "ln_final"} <= set(groups)


def test_a_leading_params_key_is_not_mistaken_for_a_module() -> None:
    tree = {"params": {"block_0": {"kernel": np.zeros((2, 2))}}}
    assert set(parameter_groups(tree)) == {"block_0"}


def test_groups_of_a_sorted_parameter_dict_are_contiguous_spans() -> None:
    params = route_params()
    groups = parameter_groups(params)
    spans = group_spans(groups)
    for name, span in spans.items():
        assert np.array_equal(np.arange(span.start, span.stop), groups[name])


def test_a_scattered_group_is_refused_rather_than_silently_fancy_indexed() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        group_spans({"scattered": np.array([0, 2, 5])})


def test_restrict_keeps_the_chosen_modules_and_zeroes_the_rest() -> None:
    params = route_params()
    groups = parameter_groups(params)
    vector = np.arange(ravel_pytree(params)[0].size, dtype=np.float64) + 1.0

    kept = restrict(vector, groups, ["block_1"])

    assert np.array_equal(kept[groups["block_1"]], vector[groups["block_1"]])
    assert np.count_nonzero(kept) == len(groups["block_1"])
    assert np.array_equal(restrict(vector, groups, "block_1"), kept)


def test_restricting_to_every_group_is_the_whole_vector() -> None:
    params = route_params()
    groups = parameter_groups(params)
    vector = np.arange(ravel_pytree(params)[0].size, dtype=np.float64) + 1.0

    assert np.array_equal(restrict(vector, groups, list(groups)), vector)


def test_an_unknown_group_is_named_in_the_error() -> None:
    groups = parameter_groups(route_params())
    with pytest.raises(ValueError, match="block_99"):
        restrict(np.zeros(1000), groups, ["block_99"])


def test_shares_divide_the_axis_and_the_parameters() -> None:
    params = route_params()
    groups = parameter_groups(params)
    size = ravel_pytree(params)[0].size
    axis = np.random.default_rng(0).normal(size=size)

    assert sum(group_shares(axis, groups).values()) == pytest.approx(1.0)
    assert sum(parameter_shares(groups, size).values()) == pytest.approx(1.0)


def test_a_random_axis_is_enriched_nowhere() -> None:
    """The baseline the campaign reads every profile against."""
    params = route_params(layers=4, width=64)
    groups = parameter_groups(params)
    size = ravel_pytree(params)[0].size
    axis = np.random.default_rng(1).normal(size=size)

    shares = group_shares(axis, groups)
    sizes = parameter_shares(groups, size)
    for name in ("block_0", "block_1", "block_2", "block_3"):
        assert shares[name] / sizes[name] == pytest.approx(1.0, abs=0.15)


def test_blocks_to_cover_counts_largest_first() -> None:
    assert blocks_to_cover({"a": 0.95, "b": 0.03, "c": 0.02}, 0.9) == 1
    assert blocks_to_cover({"a": 0.5, "b": 0.45, "c": 0.05}, 0.9) == 2
    assert blocks_to_cover({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}, 0.9) == 4
    with pytest.raises(ValueError, match="fraction"):
        blocks_to_cover({"a": 1.0}, 0.0)


def test_an_axis_planted_in_one_block_is_found_there_and_only_there() -> None:
    params = route_params(layers=3)
    groups = parameter_groups(params)
    size = ravel_pytree(params)[0].size
    rng = np.random.default_rng(2)
    offsets = np.array([-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4])

    axis = np.zeros(size)
    axis[groups["block_1"]] = rng.normal(size=len(groups["block_1"]))
    diffs = rng.normal(size=size) + np.outer(offsets, axis) + 0.02 * rng.normal(size=(len(offsets), size))

    profile = axis_by_group(offsets, diffs, groups, splits=40, seed=0)

    assert profile["block_1"]["share"] > 0.9
    assert profile["block_1"]["enrichment"] > 2.0
    assert profile["block_1"]["reliability"] > 0.9
    assert profile["block_0"]["reliability"] < 0.3
    assert profile["block_2"]["reliability"] < 0.3
    assert blocks_to_cover(group_shares(axis, groups), 0.9) == 1
