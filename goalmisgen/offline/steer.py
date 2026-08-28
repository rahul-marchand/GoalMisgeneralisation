"""Steered decoding: push or clamp the SEP residual and let the model decode.

The gap probes read a direction in the residual stream at SEP. This module
makes that direction causal or not: add a calibrated multiple of it to the
stream at a chosen depth and decode the route with everything downstream
recomputed - later blocks, their key/value cache entries at SEP, and the head.

The injection point uses the model's own depth convention: ``depth`` 0 is the
embedding and ``depth`` k is the residual after block k, matching the indexing
of :func:`goalmisgen.offline.probe.cell_residuals`. An injection at depth k is
seen by blocks k+1..n; injecting after the last block moves only the first
action's logit, which is why the experiments inject mid-stack.

Calibration: a ridge probe ``y = w . (x - mean) / std + b`` has gradient
``g = w / std`` in raw activation space, so adding ``alpha * g / |g|^2`` moves
that probe's readout by exactly ``alpha`` (steps, if the probe was trained in
steps). Whether the *model's* behaviour moves by alpha steps is the experiment,
not the construction.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.offline.decode import Decoded, decode_batch_size
from goalmisgen.offline.fast_decode import (
    NO_ACTION,
    _attend,
    _embed_prefix,
    _layer_norm,
    _logits,
    _mlp,
    _qkv,
    action_step,
)
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


def unit_dose(w: np.ndarray, std: np.ndarray) -> np.ndarray:
    """The raw-space vector that moves the probe's readout by exactly one unit."""
    g = np.asarray(w)[:-1] / np.asarray(std)
    return g / float(g @ g)


def _inject(x, delta, positions: str):
    if positions == "sep":
        return x.at[:, -1].add(delta)
    if positions == "prefix":
        return x + delta[:, None, :]
    raise ValueError(f"positions must be 'sep' or 'prefix', got {positions!r}")


def steered_prefix_pass(params, cfg: ModelConfig, observations, depth: int, delta, positions: str = "sep"):
    """:func:`fast_decode.prefix_pass` with ``delta`` added at ``depth``.

    ``delta`` is ``(batch, d_model)``. ``positions`` is ``"sep"`` (the decision
    token alone) or ``"prefix"`` (every cell token and SEP - the write a
    redundant or re-derived readout cannot route around). Everything downstream
    of the injection is recomputed, including the cache entries later blocks
    write over the prefix.
    """
    x = _embed_prefix(params, cfg, observations)
    if depth == 0:
        x = _inject(x, delta, positions)
    batch = observations.shape[0]
    head = cfg.d_model // cfg.n_heads
    shape = (cfg.n_layers, batch, cfg.sequence_length, cfg.n_heads, head)
    keys, values = jnp.zeros(shape), jnp.zeros(shape)
    for index in range(cfg.n_layers):
        block = params[f"block_{index}"]
        q, k, v = _qkv(_layer_norm(x, block["ln_attention"]), block["attention"]["qkv"], cfg.n_heads)
        keys = keys.at[index, :, : cfg.prefix_length].set(k)
        values = values.at[index, :, : cfg.prefix_length].set(v)
        x = x + _attend(q, k, v, None, block["attention"]["out"])
        x = _mlp(x, block)
        if depth == index + 1:
            x = _inject(x, delta, positions)
    return x, (keys, values)


@functools.partial(jax.jit, static_argnums=(0, 1, 4))
def _steered_decode_chunk(cfg: ModelConfig, depth: int, params, observations, positions: str, delta):
    prefix, cache = steered_prefix_pass(params, cfg, observations, depth, delta, positions)
    batch = observations.shape[0]
    first = jnp.argmax(_logits(params, prefix[:, -1:])[:, 0], axis=-1)

    def body(carry, position):
        cache, token, finished, lengths = carry
        ended = (token == cfg.eos) & ~finished
        lengths = jnp.where(ended, position, lengths)
        finished = finished | ended
        emitted = jnp.where(finished, NO_ACTION, token)
        x, cache = action_step(params, cfg, cache, jnp.where(emitted >= 0, emitted, cfg.pad), position)
        token = jnp.argmax(_logits(params, x)[:, 0], axis=-1)
        return (cache, token, finished, lengths), emitted

    init = (cache, first, jnp.zeros(batch, bool), jnp.full(batch, cfg.max_actions, jnp.int32))
    (_, last, finished, lengths), actions = jax.lax.scan(body, init, jnp.arange(cfg.max_actions))
    ended = (last == cfg.eos) & ~finished
    return actions.T, jnp.where(ended, cfg.max_actions, lengths), finished | ended


def steered_decode(
    model: RoutePrefixLM,
    params,
    observations: np.ndarray,
    depth: int,
    delta: np.ndarray,
    batch_size: int | None = None,
    positions: str = "sep",
) -> Decoded:
    """Greedy decode with a per-level steering vector added at ``depth``.

    ``delta`` is ``(n_levels, d_model)`` - a shared dose is broadcast by the
    caller. Zero delta reproduces :func:`fast_decode.greedy_decode_cached`
    exactly; the tests pin that.
    """
    if model.dtype != jnp.float32:
        raise ValueError(f"steered decoding is float32 only, got dtype={model.dtype}")
    cfg = model.config
    if not 0 <= depth <= cfg.n_layers:
        raise ValueError(f"depth must be in 0..{cfg.n_layers}, got {depth}")
    if delta.shape != (len(observations), cfg.d_model):
        raise ValueError(f"delta must be (n_levels, d_model), got {delta.shape}")
    if positions not in ("sep", "prefix"):
        raise ValueError(f"positions must be 'sep' or 'prefix', got {positions!r}")
    batch_size = decode_batch_size(model) if batch_size is None else batch_size
    tree = params["params"] if "params" in params else params

    actions, lengths, finished = [], [], []
    for start in range(0, len(observations), batch_size):
        stop = start + batch_size
        a, n, f = _steered_decode_chunk(
            cfg, depth, tree, jnp.asarray(observations[start:stop]), positions, jnp.asarray(delta[start:stop], dtype=jnp.float32)
        )
        actions.append(np.asarray(a, dtype=np.int32))
        lengths.append(np.asarray(n, dtype=np.int32))
        finished.append(np.asarray(f, dtype=bool))
    return Decoded(np.concatenate(actions), np.concatenate(lengths), np.concatenate(finished))
