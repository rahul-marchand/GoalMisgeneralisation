"""A ViT-style transformer policy for the maze, as a cleanba ``PolicySpec``.

Exists to ask whether the DRC results are properties of the task or of the one
recurrent convolutional architecture they were found on. So the network is
deliberately the plainest thing that deserves the name: one token per maze
cell, a learned two-dimensional position embedding, a stack of pre-LayerNorm
self-attention blocks, and then the same kind of readout the DRC and the ResNet
hand to cleanba's actor and critic heads. No recurrence, no state between
environment steps — every decision is a fresh function of the observation.

Three things about it are choices rather than defaults, and are recorded here
because nothing in the code would otherwise say so:

**Readout.** cleanba's ``Policy`` calls ``network_params(obs)`` and expects a
flat hidden vector. The DRC flattens its (H, W, 32) final hidden state and
passes it through ``Dense(256)``+ReLU; ``GuezResNet`` flattens its last feature
map and does the same. This network flattens its (H, W, d_model) residual
stream after a final LayerNorm and does the same, so the three architectures
differ in the trunk and nowhere else. A CLS token or a mean-pool would be more
idiomatic and would also be a second difference.

**Input scale.** ``Policy._maybe_normalize_input_image`` divides observations
by 255 because cleanba's environments emit uint8 images. Ours are already in
[0, 1], so the DRC and the ResNet were trained on inputs of order 1/255 and
Adam let them live with it. A pre-LN transformer is less forgiving: the token
embedding is summed with a position embedding before the first norm, and at a
1/255 scale the content is a rounding error on the position. ``input_scale``
undoes the division. The information reaching the network is identical; only
its units change. Set it to 1.0 to reproduce the others' conditions exactly.

**Size.** 4 layers, d_model 64, 4 heads, 4x MLP: about 200k parameters in the
trunk, against 1.2M in the shared flattened head. Small because the task is
small — 121 tokens of five channels — and because the DRC this stands beside
has a 32-channel trunk. Bigger is an experiment, not a default.

The per-block residual stream is recorded with ``sow`` as an (H, W, d_model)
grid, so the per-cell linear probes written for the DRC's hidden state apply
unchanged; :mod:`goalmisgen.nets.readers` reads it back.
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
from cleanba.network import PolicySpec

RESIDUAL = "residual"
"""Name under which each layer's residual stream is sown into ``intermediates``.

Sown once after the embedding and once after every block, in order, so the
collection holds ``n_layers + 1`` grids of shape (batch, height, width, d_model).
"""


@dataclasses.dataclass(frozen=True)
class TransformerSpec(PolicySpec):
    """Configuration for :class:`MazeTransformer`; see the module docstring."""

    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    mlp_hiddens: tuple[int, ...] = (256,)
    input_scale: float = 255.0
    pos_init_std: float = 0.02

    def make(self) -> nn.Module:
        return MazeTransformer(self)

    @property
    def n_probe_layers(self) -> int:
        """How many residual grids a forward pass records: embedding plus each block."""
        return self.n_layers + 1


class Block(nn.Module):
    """Pre-LN transformer block: ``x + Attn(LN(x))`` then ``x + MLP(LN(x))``."""

    d_model: int
    n_heads: int
    mlp_ratio: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            deterministic=True,
            name="attention",
        )(nn.LayerNorm(name="norm_attention")(x))
        x = x + attended

        hidden = nn.LayerNorm(name="norm_mlp")(x)
        hidden = nn.Dense(self.d_model * self.mlp_ratio, name="mlp_in")(hidden)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(self.d_model, name="mlp_out")(hidden)
        return x + hidden


class MazeTransformer(nn.Module):
    cfg: TransformerSpec

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        assert x.ndim == 4, f"expected NHWC observations, got shape {x.shape}"
        batch, height, width, _ = x.shape
        d_model = self.cfg.d_model

        x = x * self.cfg.input_scale
        tokens = nn.Dense(d_model, name="embed")(x)
        position = self.param("position", nn.initializers.normal(stddev=self.cfg.pos_init_std), (height, width, d_model))
        tokens = tokens + position[None]
        self.sow("intermediates", RESIDUAL, tokens)

        stream = tokens.reshape(batch, height * width, d_model)
        for index in range(self.cfg.n_layers):
            stream = Block(d_model, self.cfg.n_heads, self.cfg.mlp_ratio, name=f"block_{index}")(stream)
            self.sow("intermediates", RESIDUAL, stream.reshape(batch, height, width, d_model))

        stream = nn.LayerNorm(name="final_norm")(stream)
        out = stream.reshape(batch, -1)
        for hidden in self.cfg.mlp_hiddens:
            out = self.cfg.norm(out)
            out = nn.Dense(hidden)(out)
            out = nn.relu(out)
        return out


def residual_grids(intermediates: dict) -> tuple[jnp.ndarray, ...]:
    """The sown residual grids of one forward pass, embedding first.

    ``intermediates`` is the collection returned by
    ``policy.apply(..., mutable=["intermediates"])``; the network lives under
    ``network_params`` inside cleanba's ``Policy``.
    """
    return tuple(intermediates["network_params"][RESIDUAL])
