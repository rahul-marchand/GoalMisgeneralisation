"""Tests for the end-of-input token readout: the scalar site ``031`` adds."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline import summary
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.probe import cell_residuals
from goalmisgen.offline.train import initial_params

TINY = ModelConfig(size=7, n_channels=5, max_actions=16, d_model=32, n_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=60, seed=0, block_size=30)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=0.0, seed=0, max_actions=16)


@pytest.fixture(scope="module")
def model_and_params():
    model = RoutePrefixLM(TINY)
    return model, initial_params(model, jax.random.PRNGKey(0))


def test_sep_is_one_vector_per_episode_per_depth(demos, model_and_params):
    model, params = model_and_params
    sep = summary.sep_residuals(model, params, demos.observations(np.arange(5)), batch_size=2)
    assert sep.shape == (TINY.n_layers + 1, 5, TINY.d_model)


def test_sep_is_not_a_cell(demos, model_and_params):
    """It is the position after the last maze token, not the last maze token."""
    model, params = model_and_params
    observations = demos.observations(np.arange(4))
    sep = summary.sep_residuals(model, params, observations)
    cells = cell_residuals(model, params, observations)
    last_cell = cells[:, :, TINY.size - 1, TINY.size - 1]
    assert not np.allclose(sep, last_cell)


def test_a_scalar_probe_finds_a_linear_target_and_the_controls_fail():
    rng = np.random.default_rng(0)
    direction = rng.normal(size=16)
    train_x, test_x = rng.normal(size=(200, 16)), rng.normal(size=(200, 16))
    train_y, test_y = train_x @ direction, test_x @ direction

    found = summary.scalar_probe("residual", train_x, train_y, test_x, test_y)
    assert found.r2 > 0.95
    assert found.mae < 1.0

    shuffled = summary.scalar_probe("shuffled", train_x, train_y, test_x, rng.permutation(test_y))
    assert shuffled.r2 < 0.5


def test_a_scalar_probe_drops_episodes_with_no_target():
    """An unreachable objective must not become a number the probe can learn."""
    rng = np.random.default_rng(1)
    direction = rng.normal(size=8)
    train_x, test_x = rng.normal(size=(120, 8)), rng.normal(size=(60, 8))
    train_y, test_y = train_x @ direction, test_x @ direction
    train_y[:20] = np.nan
    test_y[:10] = np.nan

    found = summary.scalar_probe("residual", train_x, train_y, test_x, test_y)
    assert found.n == 50
    assert found.r2 > 0.95


def test_own_and_other_reports_a_paired_difference():
    """Built so the objective each episode reached is decoded more precisely."""
    rng = np.random.default_rng(2)
    n = 200
    reached = rng.integers(0, 2, n)
    truths = {feature: rng.uniform(2, 15, n) for feature in range(2)}
    predictions = {}
    for feature in range(2):
        error = np.where(reached == feature, 0.2, 3.0) * rng.normal(size=n)
        predictions[feature] = truths[feature] + error

    own, other, (low, high) = summary.own_and_other(predictions, truths, reached)
    assert own < other
    assert low > 0, "the paired interval should exclude zero when own is built to be sharper"


def test_own_and_other_is_flat_when_both_are_equally_read():
    rng = np.random.default_rng(3)
    n = 300
    reached = rng.integers(0, 2, n)
    truths = {feature: rng.uniform(2, 15, n) for feature in range(2)}
    predictions = {feature: truths[feature] + rng.normal(size=n) for feature in range(2)}

    _, _, (low, high) = summary.own_and_other(predictions, truths, reached)
    assert low < 0 < high


def test_own_and_other_skips_episodes_that_reached_nothing():
    truths = {0: np.array([3.0, 4.0]), 1: np.array([5.0, 6.0])}
    predictions = {0: np.array([3.0, 4.0]), 1: np.array([5.0, 6.0])}
    own, other, _ = summary.own_and_other(predictions, truths, np.array([-1, 0]))
    assert own == 0.0 and other == 0.0
