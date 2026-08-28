"""The cached decode must agree with the uncached one exactly.

``fast_decode`` duplicates ``RoutePrefixLM``'s forward pass to hold a key/value
cache, so the duplication is what these tests police. Both decodes take an
argmax, so agreement is exact rather than approximate: a disagreement is a
different route, not a rounding difference.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goalmisgen.offline import fast_decode
from goalmisgen.offline.decode import greedy_decode
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


@pytest.fixture(scope="module")
def model_and_params():
    cfg = ModelConfig(size=7, n_channels=4, n_layers=3, d_model=32, n_heads=4, max_actions=12)
    model = RoutePrefixLM(cfg)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((1, cfg.size, cfg.size, cfg.n_channels)), jnp.zeros((1, cfg.max_actions), jnp.int32))
    # Random parameters, not the initialisation: near-zero weights make every
    # head nearly uniform, which would hide a mis-set mask.
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(rng, len(leaves))
    params = jax.tree_util.tree_unflatten(treedef, [jax.random.normal(k, leaf.shape) * 0.5 for k, leaf in zip(keys, leaves)])
    return model, params


def observations(cfg, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, cfg.size, cfg.size, cfg.n_channels)).astype(np.float32)


def test_layer_norm_epsilon_matches_flax():
    """The manual LayerNorm hardcodes flax's default epsilon; catch a change to it."""
    assert nn.LayerNorm().epsilon == fast_decode.LN_EPS


def test_manual_forward_matches_the_module(model_and_params):
    """The duplicated forward pass reproduces the module over the prefix.

    Checked before any caching, so a mismatch here is a transcription error in
    the block rather than a mistake about what the cache may reuse.
    """
    model, params = model_and_params
    cfg = model.config
    obs = observations(cfg, 4)
    actions = np.full((4, cfg.max_actions), -1, dtype=np.int32)

    _, residuals = model.apply(params, jnp.asarray(obs), jnp.asarray(actions))
    manual, _ = fast_decode.prefix_pass(params["params"], cfg, jnp.asarray(obs))

    # The prefix never attends to an action, so the module's residual at those
    # positions must equal the prefix-only pass whatever the actions are.
    #
    # Agreement here is float32 reassociation, not exactness: the same sums are
    # taken in a different order, so the bound scales with the activations rather
    # than being absolute. The tight test is the decode below, which compares
    # argmaxes and must match element for element.
    expected = np.asarray(residuals[-1][:, : cfg.prefix_length])
    difference = np.abs(np.asarray(manual) - expected).max()
    assert difference < 1e-5 * max(np.abs(expected).max(), 1.0), difference


def test_prefix_is_independent_of_the_actions(model_and_params):
    """The premise the cache rests on, asserted rather than assumed."""
    model, params = model_and_params
    cfg = model.config
    obs = jnp.asarray(observations(cfg, 3))
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.integers(0, cfg.n_actions, size=(3, cfg.max_actions)).astype(np.int32))
    b = jnp.asarray(rng.integers(0, cfg.n_actions, size=(3, cfg.max_actions)).astype(np.int32))

    _, ra = model.apply(params, obs, a)
    _, rb = model.apply(params, obs, b)
    np.testing.assert_array_equal(np.asarray(ra[-1][:, : cfg.prefix_length]), np.asarray(rb[-1][:, : cfg.prefix_length]))


def test_cached_decode_matches_greedy_decode(model_and_params):
    model, params = model_and_params
    obs = observations(model.config, 24, seed=3)
    slow = greedy_decode(model, params, obs)
    fast = fast_decode.greedy_decode_cached(model, params, obs)

    np.testing.assert_array_equal(fast.actions, slow.actions)
    np.testing.assert_array_equal(fast.lengths, slow.lengths)
    np.testing.assert_array_equal(fast.emitted_eos, slow.emitted_eos)


def test_cached_decode_is_independent_of_batching(model_and_params):
    """``batch_size`` chunks the work and must not change the answer."""
    model, params = model_and_params
    obs = observations(model.config, 20, seed=5)
    whole = fast_decode.greedy_decode_cached(model, params, obs, batch_size=64)
    split = fast_decode.greedy_decode_cached(model, params, obs, batch_size=7)
    np.testing.assert_array_equal(whole.actions, split.actions)
    np.testing.assert_array_equal(whole.lengths, split.lengths)


def test_narrow_dtype_is_refused(model_and_params):
    """Silently ignoring the dtype would make the cache fast and wrong."""
    _, params = model_and_params
    cfg = ModelConfig(size=7, n_channels=4, n_layers=3, d_model=32, n_heads=4, max_actions=12)
    with pytest.raises(ValueError, match="float32 only"):
        fast_decode.greedy_decode_cached(RoutePrefixLM(cfg, dtype=jnp.bfloat16), params, observations(cfg, 2))
