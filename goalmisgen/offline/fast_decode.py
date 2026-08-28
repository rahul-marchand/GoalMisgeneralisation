"""Greedy decoding with a key/value cache.

:func:`goalmisgen.offline.decode.greedy_decode` re-runs the whole sequence at
every one of ``max_actions + 1`` steps. That is the honest thing to write and it
was the right call while decoding happened once per checkpoint, but a sweep
decodes dozens of models over tens of thousands of levels and the redundancy
dominates: 65 passes over 186 tokens where one pass over 122 plus 64
single-token steps would do.

The prefix is what makes the cache simple here. Attention is bidirectional over
the cells and SEP and causal over the actions, so a prefix position never
attends to an action (:func:`~goalmisgen.offline.model.prefix_mask`) and the
prefix's keys and values are a function of the maze alone. They are computed
once and every action step reads them.

This duplicates the forward pass of
:class:`~goalmisgen.offline.model.RoutePrefixLM` rather than restructuring it,
so the training path is untouched. ``tests/test_fast_decode.py`` pins the
duplication down: the manual forward is checked against ``model.apply``, and the
decode against ``greedy_decode`` on real demonstrations, where the routes must
be *equal* rather than close -- both take an argmax, so a real disagreement is a
changed route rather than a rounding difference.

Float32 only. The mixed-precision path exists for training the wide end of the
scaling grid and is not used for decoding; a narrower dtype is refused rather
than ignored, because ignoring it would make this fast and wrong.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.offline.decode import Decoded, decode_batch_size
from goalmisgen.offline.demos import NO_ACTION
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM

LN_EPS = 1e-6
"""``flax.linen.LayerNorm``'s default epsilon; asserted against in the tests."""

PRECISION = jax.lax.Precision.HIGHEST
"""Every matmul here is float32, not TF32.

On an Ampere-or-later card JAX defaults to TF32, whose ten mantissa bits give
about 1e-3 relative error. That is enough to change a route: against the
uncached decode at the default precision, 7 of 4000 routes came out different,
at logit margins of 0.065 -- five orders of magnitude above float32 noise, so
not a near-tie being broken the other way but the two shapes of matmul rounding
differently. At HIGHEST the two agree on all 4000.

It is close to free, because a cached step multiplies one token at a time and is
nowhere near the tensor cores' regime: 2.66s against 2.62s per 4000 levels, a 2%
cost for exactness. Note this makes the *cached* path the reproducible one --
``greedy_decode`` still runs at whatever the caller's default is, so its GPU
results are not bit-reproducible against CPU.
"""


def _layer_norm(x, p):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + LN_EPS) * p["scale"] + p["bias"]


def _dense(x, p):
    return jnp.matmul(x, p["kernel"], precision=PRECISION) + p["bias"]


def _qkv(x, p, n_heads):
    batch, length, width = x.shape
    out = _dense(x, p).reshape(batch, length, 3, n_heads, width // n_heads)
    return out[:, :, 0], out[:, :, 1], out[:, :, 2]


def _attend(q, k, v, visible, p_out):
    """``visible`` is a ``(k_len,)`` bool of which keys this query may read, or None."""
    batch, q_len, n_heads, head = q.shape
    scores = jnp.einsum("blhd,bmhd->bhlm", q, k, precision=PRECISION) / math.sqrt(head)
    if visible is not None:
        scores = jnp.where(visible[None, None, None, :], scores, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhlm,bmhd->blhd", weights, v, precision=PRECISION).reshape(batch, q_len, n_heads * head)
    return _dense(out, p_out)


def _mlp(x, block):
    h = _layer_norm(x, block["ln_mlp"])
    h = _dense(h, block["mlp_in"])
    h = jax.nn.gelu(h)
    return x + _dense(h, block["mlp_out"])


def _embed_prefix(params, cfg: ModelConfig, observations):
    batch = observations.shape[0]
    cells = observations.reshape(batch, cfg.n_cells, cfg.n_channels)
    rows = jnp.arange(cfg.n_cells) // cfg.size
    cols = jnp.arange(cfg.n_cells) % cfg.size
    x = _dense(cells, params["cell_in"]) + params["row_embedding"][rows] + params["col_embedding"][cols]
    sep = jnp.broadcast_to(params["sep"], (batch, 1, cfg.d_model))
    return jnp.concatenate([x, sep], axis=1)


def _logits(params, x):
    return _dense(_layer_norm(x, params["ln_final"]), params["head"])


def prefix_pass(params, cfg: ModelConfig, observations):
    """Every block over the prefix, returning its output and a full-length cache.

    The cache is allocated for the whole sequence and filled only over the
    prefix; action steps write into their own slot. Shapes stay static, which is
    what lets the decode be one ``jax.lax.scan`` rather than an unrolled loop.
    """
    x = _embed_prefix(params, cfg, observations)
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
    return x, (keys, values)


def action_step(params, cfg: ModelConfig, cache, token, position):
    """One action token through every block, writing into its cache slot."""
    keys, values = cache
    slot = cfg.prefix_length + position
    visible = jnp.arange(cfg.sequence_length) <= slot
    x = params["action_embedding"]["embedding"][token][:, None] + params["action_position"][position][None, None]
    for index in range(cfg.n_layers):
        block = params[f"block_{index}"]
        q, k, v = _qkv(_layer_norm(x, block["ln_attention"]), block["attention"]["qkv"], cfg.n_heads)
        keys = keys.at[index, :, slot].set(k[:, 0])
        values = values.at[index, :, slot].set(v[:, 0])
        x = x + _attend(q, keys[index], values[index], visible, block["attention"]["out"])
        x = _mlp(x, block)
    return x, (keys, values)


@functools.partial(jax.jit, static_argnums=(0,))
def _decode_chunk(cfg: ModelConfig, params, observations):
    """The unrolled loop of ``greedy_decode``, step for step, over a cache."""
    prefix, cache = prefix_pass(params, cfg, observations)
    batch = observations.shape[0]
    # Position 0's prediction comes from SEP, as in the uncached decode.
    first = jnp.argmax(_logits(params, prefix[:, -1:])[:, 0], axis=-1)

    def body(carry, position):
        cache, token, finished, lengths = carry
        ended = (token == cfg.eos) & ~finished
        lengths = jnp.where(ended, position, lengths)
        finished = finished | ended
        emitted = jnp.where(finished, NO_ACTION, token)
        # The module maps NO_ACTION to `pad` before embedding; do the same, so a
        # finished route feeds back exactly what the uncached loop feeds back.
        x, cache = action_step(params, cfg, cache, jnp.where(emitted >= 0, emitted, cfg.pad), position)
        token = jnp.argmax(_logits(params, x)[:, 0], axis=-1)
        return (cache, token, finished, lengths), emitted

    init = (cache, first, jnp.zeros(batch, bool), jnp.full(batch, cfg.max_actions, jnp.int32))
    (_, last, finished, lengths), actions = jax.lax.scan(body, init, jnp.arange(cfg.max_actions))
    # One more prediction than there are slots: a route may end exactly at the cap.
    ended = (last == cfg.eos) & ~finished
    return actions.T, jnp.where(ended, cfg.max_actions, lengths), finished | ended


def greedy_decode_cached(model: RoutePrefixLM, params, observations: np.ndarray, batch_size: int | None = None) -> Decoded:
    """Drop-in replacement for :func:`~goalmisgen.offline.decode.greedy_decode`."""
    if model.dtype != jnp.float32:
        raise ValueError(f"cached decoding is float32 only, got dtype={model.dtype}")
    cfg = model.config
    batch_size = decode_batch_size(model) if batch_size is None else batch_size
    tree = params["params"] if "params" in params else params

    actions, lengths, finished = [], [], []
    for start in range(0, len(observations), batch_size):
        chunk = jnp.asarray(observations[start : start + batch_size])
        a, n, f = _decode_chunk(cfg, tree, chunk)
        actions.append(np.asarray(a, dtype=np.int32))
        lengths.append(np.asarray(n, dtype=np.int32))
        finished.append(np.asarray(f, dtype=bool))
    return Decoded(np.concatenate(actions), np.concatenate(lengths), np.concatenate(finished))
