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


def test_a_sep_edit_touches_only_the_sep_position(demos, model_and_params):
    """The write ``032`` makes: one vector at the end-of-input token, nothing at the cells."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    module = __import__("032_bc_write_the_distance")

    direction = np.arange(TINY.d_model, dtype=float)
    grid = module.sep_edit(direction, alpha=2.0, n_episodes=3, n_cells=TINY.n_cells)
    assert grid.shape == (3, TINY.n_cells + 1, TINY.d_model)
    np.testing.assert_array_equal(grid[:, : TINY.n_cells], 0.0)
    np.testing.assert_allclose(grid[:, TINY.n_cells], np.tile(2.0 * direction, (3, 1)))


def test_a_calibrated_write_moves_the_decoded_distance_by_what_it_says(demos, model_and_params):
    """The arithmetic the whole of ``032`` rests on, end to end on real residuals.

    A probe's weight vector is not a direction in the space the activations live
    in - it is fitted on standardised inputs - and getting that wrong produces a
    plausible slope rather than a crash, which is why ``steering.verify`` exists
    and why this test reads the write back through the probe itself.
    """
    from goalmisgen.analysis import steering
    from goalmisgen.analysis.probes import apply_linear

    model, params = model_and_params
    observations = demos.observations(np.arange(40))
    sep = summary.sep_residuals(model, params, observations)[1]
    rng = np.random.default_rng(0)
    truth = sep @ rng.normal(size=TINY.d_model) * 0.1 + 7.0

    weights, mean, std = summary.fit_scalar(sep, truth)
    direction = steering.from_probe("d", weights, std)
    before = apply_linear(sep, weights, mean, std)
    after = apply_linear(sep + 3.0 * direction.delta, weights, mean, std)
    np.testing.assert_allclose(after - before, 3.0, atol=1e-6)
