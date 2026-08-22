"""A small prefix-LM transformer that reads a maze and writes a route.

The sequence is ``[cell tokens, in raster order] [SEP] [a_1 ... a_T] [EOS]``.
Each cell token is a linear map of that cell's observation channels plus a
learned row and column embedding; SEP is a learned vector; actions are
embedded with a learned position. Attention is **bidirectional over the cells
and SEP and causal over the actions** - a prefix LM - and the loss is taken on
the action positions only.

Why prefix rather than fully causal: the question this model exists to answer
is whether the route is linearly present in the residual stream *before the
first action token*, at each cell, the way it is in the DRC's recurrent state
at t=0. With bidirectional attention over the cells, the per-cell residual at
the maze positions is a function of the maze alone - the action tokens cannot
reach it - so "probe the residual at cell (r, c)" has exactly the meaning the
DRC probe has. A fully causal variant is a follow-up, not built here.

The model is deliberately small. The task is an 11x11 maze with two goals; a
four-layer, 128-wide decoder learns it in minutes on one GPU and leaves room
to train three seeds per condition.
"""

from __future__ import annotations

import dataclasses
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    size: int = 11
    """Side of the (padded) maze; the number of cell tokens is ``size**2``."""

    n_channels: int = 5
    """Observation channels per cell: wall, agent, one per colour, value."""

    n_actions: int = 4
    max_actions: int = 64
    """Longest route the model can emit. Must cover the demonstrations."""

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}")

    @property
    def n_cells(self) -> int:
        return self.size * self.size

    @property
    def prefix_length(self) -> int:
        """Cells plus SEP - the positions whose attention is bidirectional."""
        return self.n_cells + 1

    @property
    def sequence_length(self) -> int:
        return self.prefix_length + self.max_actions

    @property
    def eos(self) -> int:
        """Output class and input token for end-of-route."""
        return self.n_actions

    @property
    def pad(self) -> int:
        """Input token standing in for ``NO_ACTION`` past the end of a route."""
        return self.n_actions + 1

    @property
    def n_classes(self) -> int:
        """Moves plus EOS - what the head predicts."""
        return self.n_actions + 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{field.name: data[field.name] for field in dataclasses.fields(cls)})


def prefix_mask(prefix_length: int, sequence_length: int) -> np.ndarray:
    """``(L, L)`` bool: may position ``i`` attend to position ``j``?

    Every position sees the whole prefix; positions past it also see earlier
    action tokens and themselves. A prefix position therefore never sees an
    action, which is what keeps the per-cell residual a function of the maze.
    """
    i = np.arange(sequence_length)[:, None]
    j = np.arange(sequence_length)[None, :]
    return (j < prefix_length) | (j <= i)


class SelfAttention(nn.Module):
    n_heads: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        batch, length, width = x.shape
        head = width // self.n_heads
        dense = dict(dtype=self.dtype, param_dtype=jnp.float32)
        qkv = nn.Dense(3 * width, name="qkv", **dense)(x).reshape(batch, length, 3, self.n_heads, head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scores = jnp.einsum("blhd,bmhd->bhlm", q, k) / math.sqrt(head)
        # The softmax is taken in float32 whatever the matmuls run in. It is the
        # one place a narrow exponent bites: the masked entries are set to the
        # dtype's most negative value, and in bfloat16 that is -3.4e38 with three
        # significant digits, so exp() of the shifted scores loses the tail that
        # separates a nearly-uniform attention from a peaked one.
        scores = jnp.where(mask[None, None], scores.astype(jnp.float32), jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(scores, axis=-1).astype(self.dtype)
        out = jnp.einsum("bhlm,bmhd->blhd", weights, v).reshape(batch, length, width)
        return nn.Dense(width, name="out", **dense)(out)


class Block(nn.Module):
    """Pre-LN transformer block."""

    n_heads: int
    mlp_ratio: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        width = x.shape[-1]
        dense = dict(dtype=self.dtype, param_dtype=jnp.float32)
        # LayerNorm stays in float32: it divides by a standard deviation computed
        # over the width, and doing that in bfloat16 is the other classic way a
        # mixed-precision transformer goes quietly wrong.
        norm = dict(dtype=jnp.float32, param_dtype=jnp.float32)
        x = x + SelfAttention(self.n_heads, self.dtype, name="attention")(
            nn.LayerNorm(name="ln_attention", **norm)(x).astype(self.dtype), mask
        )
        h = nn.LayerNorm(name="ln_mlp", **norm)(x).astype(self.dtype)
        h = nn.Dense(self.mlp_ratio * width, name="mlp_in", **dense)(h)
        h = jax.nn.gelu(h)
        h = nn.Dense(width, name="mlp_out", **dense)(h)
        return x + h.astype(x.dtype)


class RoutePrefixLM(nn.Module):
    """Maze in, route out. Returns logits and the residual stream at every depth."""

    config: ModelConfig

    dtype: jnp.dtype = jnp.float32
    """Precision the matmuls run in. Parameters stay float32 whatever this says.

    Default float32, so nothing changes unless a caller asks. ``jnp.bfloat16``
    turns on the H100's tensor cores, whose peak is roughly twice what float32
    reaches through TF32 -- the probe in ``results/scaling-shape-probe-h100.txt``
    measured the widest cell of the grid at 69 TFLOP/s, about 7% of the card.

    Mixed, not narrow: weights, the LayerNorms and the attention softmax are all
    float32, and only the large matmuls drop to bfloat16. Those three exclusions
    are where a narrow exponent actually costs something; the matmuls are where
    the time goes.

    **It changes the noise floor**, so it is not free for this campaign in the
    way it is for ordinary training. Every quantity here is a comparison of
    fine-tuning signal against fine-tuning noise, so a grid trained at one
    precision cannot be compared arm-for-arm with one trained at another. Pick
    one for the whole grid.
    """

    remat: bool = True
    """Recompute each block's internals during the backward pass instead of storing them.

    Gradient checkpointing, and it is what makes the wide-and-deep end of the
    scaling grid trainable at a useful batch size. The block materialises the
    full ``(batch, heads, length, length)`` attention matrix, and that one tensor
    is the entire memory bill: at ``d_model=512``, 16 layers and batch 1024 the
    stored activations come to about 161 GB, which fits on no single card.
    Rematerialising leaves one block's peak live plus the residual stream between
    blocks, around 16 GB -- a factor of ten, for one extra forward pass.

    Not an approximation. The parameter tree is untouched, so a checkpoint
    written with this on loads with it off, and the logits are bit-identical;
    the gradients agree to float32 reassociation noise (measured at 2e-7
    relative, against an eps of 1.2e-7) because the recomputed forward pass
    sums in a different order, not because a different quantity is computed.

    Costed at roughly a third more compute (forward, forward again, backward,
    against forward and backward). It is left **on at every size**, including the
    small models that do not need it, so that a step costs the same across the
    grid and wall-clock comparisons between shapes mean something. Turning it off
    is for measuring that cost, not for going faster.
    """

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        """
        ``observations``: ``(B, size, size, n_channels)`` float.
        ``actions``: ``(B, max_actions)`` int, moves with ``-1`` past the end.

        Returns ``logits`` of shape ``(B, max_actions + 1, n_classes)`` - the
        prediction made at SEP and at each action position, i.e. for
        ``a_1 .. a_T`` and then EOS - and ``residuals``, a list of
        ``n_layers + 1`` arrays of shape ``(B, L, d_model)``: the embedding and
        the stream after each block. Slice ``[:, :n_cells]`` for the per-cell
        grid the probes read.
        """
        cfg = self.config
        batch = observations.shape[0]
        width = cfg.d_model
        small = nn.initializers.normal(stddev=0.02)

        cells = observations.reshape(batch, cfg.n_cells, cfg.n_channels)
        rows = jnp.arange(cfg.n_cells) // cfg.size
        cols = jnp.arange(cfg.n_cells) % cfg.size
        row_embedding = self.param("row_embedding", small, (cfg.size, width))
        col_embedding = self.param("col_embedding", small, (cfg.size, width))
        x_cells = nn.Dense(width, name="cell_in")(cells) + row_embedding[rows] + col_embedding[cols]

        sep = jnp.broadcast_to(self.param("sep", small, (1, 1, width)), (batch, 1, width))

        tokens = jnp.where(actions >= 0, actions, cfg.pad)
        action_embedding = nn.Embed(cfg.n_actions + 2, width, embedding_init=small, name="action_embedding")
        action_position = self.param("action_position", small, (cfg.max_actions, width))
        x_actions = action_embedding(tokens) + action_position[None, : actions.shape[1]]

        x = jnp.concatenate([x_cells, sep, x_actions], axis=1)
        mask = jnp.asarray(prefix_mask(cfg.prefix_length, x.shape[1]))

        block = nn.remat(Block) if self.remat else Block
        residuals = [x]
        for index in range(cfg.n_layers):
            x = block(cfg.n_heads, cfg.mlp_ratio, self.dtype, name=f"block_{index}")(x, mask)
            residuals.append(x)

        x = nn.LayerNorm(name="ln_final", dtype=jnp.float32, param_dtype=jnp.float32)(x)
        logits = nn.Dense(cfg.n_classes, name="head", dtype=jnp.float32, param_dtype=jnp.float32)(
            x[:, cfg.prefix_length - 1 :]
        )
        return logits, residuals


def targets_from_routes(actions: jnp.ndarray, lengths: jnp.ndarray, eos: int) -> jnp.ndarray:
    """What the head should say at each of its ``max_actions + 1`` positions.

    Position ``t`` (0-based, from SEP) predicts ``a_{t+1}``; position
    ``length`` predicts EOS; everything after is ``-1`` and carries no loss.
    """
    batch, max_actions = actions.shape
    padded = jnp.concatenate([actions, jnp.full((batch, 1), -1, dtype=actions.dtype)], axis=1)
    positions = jnp.arange(max_actions + 1)[None, :]
    return jnp.where(positions == lengths[:, None], eos, padded)


def cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean token cross-entropy over positions whose target is not ``-1``."""
    mask = targets >= 0
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, jnp.maximum(targets, 0)[..., None], axis=-1)[..., 0]
    return -(picked * mask).sum() / jnp.maximum(mask.sum(), 1)


def token_accuracy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    mask = targets >= 0
    correct = (logits.argmax(-1) == targets) & mask
    return correct.sum() / jnp.maximum(mask.sum(), 1)


def parameter_count(params) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params)))
