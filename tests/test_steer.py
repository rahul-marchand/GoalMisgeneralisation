"""Tests for steered decoding of the route model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goalmisgen.offline.fast_decode import greedy_decode_cached, prefix_pass
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.steer import steered_decode, steered_prefix_pass, unit_dose
from goalmisgen.offline.train import initial_params

TINY = ModelConfig(size=7, n_channels=4, max_actions=16, d_model=32, n_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def setup():
    model = RoutePrefixLM(TINY)
    params = initial_params(model, jax.random.PRNGKey(0))
    observations = np.asarray(
        np.random.default_rng(0).random((12, TINY.size, TINY.size, TINY.n_channels)), dtype=np.float32
    )
    return model, params, observations


def test_zero_delta_reproduces_the_cached_decode(setup):
    model, params, observations = setup
    zero = np.zeros((len(observations), TINY.d_model), dtype=np.float32)
    for depth in range(TINY.n_layers + 1):
        steered = steered_decode(model, params, observations, depth, zero, batch_size=5)
        plain = greedy_decode_cached(model, params, observations, batch_size=5)
        np.testing.assert_array_equal(steered.actions, plain.actions)
        np.testing.assert_array_equal(steered.lengths, plain.lengths)


def test_injection_after_the_last_block_is_exactly_additive(setup):
    model, params, observations = setup
    tree = params["params"]
    delta = np.asarray(np.random.default_rng(1).normal(size=(len(observations), TINY.d_model)), dtype=np.float32)
    plain, _ = prefix_pass(tree, TINY, jnp.asarray(observations))
    steered, _ = steered_prefix_pass(tree, TINY, jnp.asarray(observations), TINY.n_layers, jnp.asarray(delta))
    np.testing.assert_allclose(np.asarray(steered[:, -1]), np.asarray(plain[:, -1]) + delta, atol=1e-5)
    # Cell positions are untouched by a SEP injection.
    np.testing.assert_allclose(np.asarray(steered[:, :-1]), np.asarray(plain[:, :-1]), atol=1e-6)


def test_mid_stack_injection_changes_downstream_only(setup):
    model, params, observations = setup
    tree = params["params"]
    delta = np.asarray(np.random.default_rng(2).normal(size=(len(observations), TINY.d_model)), dtype=np.float32)
    plain, (pk, pv) = prefix_pass(tree, TINY, jnp.asarray(observations))
    steered, (sk, sv) = steered_prefix_pass(tree, TINY, jnp.asarray(observations), 1, jnp.asarray(delta))
    # Block 0 ran before the injection: its cache entries agree.
    np.testing.assert_allclose(np.asarray(sk[0]), np.asarray(pk[0]), atol=1e-6)
    # Block 1 ran after: its SEP cache entry differs, and so does the output.
    assert not np.allclose(np.asarray(sk[1][:, TINY.prefix_length - 1]), np.asarray(pk[1][:, TINY.prefix_length - 1]))
    assert not np.allclose(np.asarray(steered[:, -1]), np.asarray(plain[:, -1]))


def test_unit_dose_moves_the_readout_by_one():
    rng = np.random.default_rng(0)
    w = rng.normal(size=9)  # 8 weights + bias, as fit_ridge returns
    std = rng.uniform(0.5, 2.0, size=8)
    mean = rng.normal(size=8)
    u = unit_dose(w, std)
    x = rng.normal(size=8)

    def read(x):
        return float(w[:-1] @ ((x - mean) / std) + w[-1])

    assert read(x + 3.0 * u) - read(x) == pytest.approx(3.0, abs=1e-9)


def test_prefix_wide_injection(setup):
    model, params, observations = setup
    tree = params["params"]
    zero = np.zeros((len(observations), TINY.d_model), dtype=np.float32)
    steered = steered_decode(model, params, observations, 1, zero, batch_size=5, positions="prefix")
    plain = greedy_decode_cached(model, params, observations, batch_size=5)
    np.testing.assert_array_equal(steered.actions, plain.actions)

    delta = np.asarray(np.random.default_rng(3).normal(size=(len(observations), TINY.d_model)), dtype=np.float32)
    plain_x, _ = prefix_pass(tree, TINY, jnp.asarray(observations))
    steered_x, _ = steered_prefix_pass(tree, TINY, jnp.asarray(observations), TINY.n_layers, jnp.asarray(delta), "prefix")
    np.testing.assert_allclose(np.asarray(steered_x), np.asarray(plain_x) + delta[:, None, :], atol=1e-5)


def test_shape_and_depth_validation(setup):
    model, params, observations = setup
    with pytest.raises(ValueError):
        steered_decode(model, params, observations, TINY.n_layers + 1, np.zeros((len(observations), TINY.d_model)))
    with pytest.raises(ValueError):
        steered_decode(model, params, observations, 0, np.zeros((3, TINY.d_model)))
