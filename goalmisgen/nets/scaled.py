"""Undo cleanba's division by 255 in front of a network that cannot absorb it.

``Policy._maybe_normalize_input_image`` divides every observation by 255
because cleanba's environments emit uint8 images. Ours are already in [0, 1],
so a network behind it sees inputs of order 1/255. The DRC lives with that:
its gates are sigmoids and tanhs, so its hidden state is O(1) whatever the
input scale, and Adam grows the input weights. A plain ReLU ResNet with no
normalisation does not: every activation, and so every logit and value, is
proportional to the input scale, the policy stays uniform and the critic
cannot move off its bias — resnet11's entropy sat at ln 4 and its value loss
at its initial level for 25M steps while the DRC had taken off at 15M and the
transformer, which rescales its own input, at 12M.

The remedy is the one the transformer already applies, lifted out so any
spec can have it: multiply the observation back up before the wrapped
network sees it. cleanba's code is untouched; the wrapped network is
constructed by its own ``make`` and is exactly the module it would have been,
one multiplication earlier in the graph. The information reaching it is
identical; only the units change.
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax
from cleanba.network import GuezResNetConfig, PolicySpec


@dataclasses.dataclass(frozen=True)
class ScaledInputSpec(PolicySpec):
    """``inner`` applied to ``scale * obs``. Head settings come from this spec, as cleanba's ``Policy`` expects."""

    inner: PolicySpec = GuezResNetConfig()
    scale: float = 255.0

    def make(self) -> nn.Module:
        return ScaledInput(self)


class ScaledInput(nn.Module):
    cfg: ScaledInputSpec

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.cfg.inner.make()(x * self.cfg.scale)
