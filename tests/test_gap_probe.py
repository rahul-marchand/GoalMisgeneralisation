"""Tests for the per-level distance-gap probes on the route model."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.gap_probe import (
    SITES,
    collect_site_features,
    fit_gap_probe,
    flatten_depths,
    gap_targets,
    within_cell_choice,
)
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.probe import cell_residuals
from goalmisgen.offline.train import initial_params

TINY = ModelConfig(size=7, n_channels=5, max_actions=16, d_model=32, n_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=80, seed=0, block_size=40)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16)


@pytest.fixture(scope="module")
def model_and_params():
    model = RoutePrefixLM(TINY)
    return model, initial_params(model, jax.random.PRNGKey(0))


def test_targets_come_from_the_solver(demos):
    targets = gap_targets(demos, np.arange(len(demos)))
    distances = np.asarray(demos.distances)
    values = np.asarray(demos.values)
    for row in range(len(demos)):
        richer = int(np.argmax(values[row]))
        assert targets.richer[row] == richer
        assert targets.d_rich[row] == distances[row, richer]
        assert targets.d_poor[row] == distances[row, 1 - richer]
    assert np.array_equal(targets.gap, targets.d_rich - targets.d_poor)
    assert np.array_equal(targets.valid, (targets.d_rich >= 0) & (targets.d_poor >= 0))


def test_every_site_yields_one_vector_per_level_per_depth(demos, model_and_params):
    model, params = model_and_params
    features = collect_site_features(model, params, demos, np.arange(10), batch_size=4)
    assert set(features) == set(SITES)
    depths = TINY.n_layers + 1
    assert features["sep"].shape == (10, depths, TINY.d_model)
    assert features["agent"].shape == (10, depths, TINY.d_model)
    assert features["objectives"].shape == (10, depths, 2 * TINY.d_model)
    assert features["cells_mean"].shape == (10, depths, TINY.d_model)
    assert flatten_depths(features["sep"], layer=1).shape == (10, TINY.d_model)
    assert flatten_depths(features["sep"], layer=None).shape == (10, depths * TINY.d_model)


def test_sites_agree_with_the_cell_capture(demos, model_and_params):
    """The mean site is the mean of the per-cell grid; SEP is what that capture drops."""
    model, params = model_and_params
    indices = np.arange(6)
    features = collect_site_features(model, params, demos, indices, batch_size=3)
    grids = cell_residuals(model, params, demos.observations(indices), batch_size=3)
    np.testing.assert_allclose(
        features["cells_mean"],
        grids.reshape(grids.shape[0], len(indices), -1, TINY.d_model).mean(axis=2).transpose(1, 0, 2),
        atol=1e-5,
    )
    agent = np.asarray(demos.agent)[indices]
    for row, (r, c) in enumerate(agent):
        np.testing.assert_allclose(features["agent"][row, 0], grids[0, row, r, c], atol=1e-6)


def test_chunking_does_not_change_the_features(demos, model_and_params):
    model, params = model_and_params
    whole = collect_site_features(model, params, demos, np.arange(9), batch_size=64)
    split = collect_site_features(model, params, demos, np.arange(9), batch_size=2)
    for site in SITES:
        np.testing.assert_allclose(whole[site], split[site], atol=1e-5)


def test_probe_recovers_a_planted_target():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 8))
    y = 3.0 * x[:, 2] - x[:, 5] + 7.0
    result = fit_gap_probe(x[:3000], y[:3000], x[3000:], y[3000:])
    assert result.r2 > 0.99
    assert result.mae < 0.1

    shuffled = fit_gap_probe(x[:3000], rng.permutation(y[:3000]), x[3000:], y[3000:])
    assert abs(shuffled.r2) < 0.05


def test_within_cell_choice_scores_signal_and_noise():
    rng = np.random.default_rng(0)
    n = 2000
    d_rich = rng.integers(10, 13, size=n)
    d_poor = rng.integers(2, 4, size=n)
    misread = rng.normal(size=n)
    choice = misread < 0.0  # the model takes the richer objective when it under-reads the gap

    informed = within_cell_choice(d_rich, d_poor, score=misread, choice=choice)
    assert informed.auc > 0.95
    assert informed.r2 > 0.5
    assert informed.n_levels == n

    blind = within_cell_choice(d_rich, d_poor, score=rng.normal(size=n), choice=choice)
    assert abs(blind.auc - 0.5) < 0.1
    assert blind.r2 < 0.05

    sparse = within_cell_choice(d_rich, d_poor, score=misread, choice=choice, min_n=10_000)
    assert sparse.n_cells == 0 and np.isnan(sparse.auc)
