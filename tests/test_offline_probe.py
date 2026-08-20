"""Tests for reading the route model's residual stream into the probe machinery."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from goalmisgen.analysis.probes import probe
from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
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


def test_residuals_have_one_grid_per_depth(demos, model_and_params):
    model, params = model_and_params
    streams = cell_residuals(model, params, demos.observations(np.arange(5)), batch_size=2)
    assert streams.shape == (TINY.n_layers + 1, 5, 7, 7, TINY.d_model)


def test_capture_yields_what_the_probes_read(demos, model_and_params):
    model, params = model_and_params
    rollouts = capture(model, params, demos, np.arange(20), layer=1)
    assert len(rollouts) == 20
    r = rollouts[0]
    assert r.features.shape == (7, 7, TINY.d_model)
    assert r.observation.shape == (7, 7, 5)
    assert r.visited.dtype == bool and r.visited[r.level.agent_start]
    assert r.visit_step[r.level.agent_start] == 0
    assert r.distance[r.level.agent_start] == 0
    # The probe machinery accepts them as-is.
    result = probe(rollouts[:12], rollouts[12:], source="observation")
    assert 0.0 <= result.auc <= 1.0


def test_all_depths_concatenate(demos, model_and_params):
    model, params = model_and_params
    rollouts = capture(model, params, demos, np.arange(3), layer=None)
    assert rollouts[0].features.shape == (7, 7, (TINY.n_layers + 1) * TINY.d_model)
    last = capture(model, params, demos, np.arange(3), layer=TINY.n_layers)
    np.testing.assert_array_equal(rollouts[0].features[..., -TINY.d_model :], last[0].features)


def test_reader_params_change_features_but_not_labels(demos, model_and_params):
    model, params = model_and_params
    other = initial_params(model, jax.random.PRNGKey(1))
    a = capture(model, params, demos, np.arange(6), layer=2)
    b = capture(model, params, demos, np.arange(6), layer=2, reader_params=other)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.visited, y.visited)
        assert not np.allclose(x.features, y.features)


def test_relabel_walks_to_the_named_objective(demos, model_and_params):
    model, params = model_and_params
    rollouts = capture(model, params, demos, np.arange(10), layer=0)
    optimal = relabel(rollouts, "optimal")
    feature0 = relabel(rollouts, "feature0")
    for r, o, f in zip(rollouts, optimal, feature0):
        level = r.level
        best = level.objectives[r.info["optimal_index"]].position
        assert o.visited[best] and o.visit_step[best] == r.info["optimal_distance"]
        colour0 = next(ob.position for ob in level.objectives if ob.feature_id == 0)
        assert f.visited[colour0]
        # At rho=0 colour 0 marks the poorer objective, so the two routes differ whenever the expert takes the richer one.
        if r.info["optimal_index"] != next(k for k, ob in enumerate(level.objectives) if ob.feature_id == 0):
            assert not np.array_equal(o.visited, f.visited)
        np.testing.assert_array_equal(o.features, r.features)
    with pytest.raises(ValueError):
        relabel(rollouts, "nearest")
