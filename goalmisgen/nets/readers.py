"""Read the per-cell state off one forward pass, whatever the architecture.

The plan probes were written against the DRC, whose working state is its
carry: three ConvLSTM layers, each a 32-channel vector per maze cell. A
non-recurrent network has no carry, but it has the same kind of object — a
feature map, or a residual stream, with a vector per cell — and the probes do
not care which, provided someone hands them ``(height, width, channels)`` grids
with the layer boundaries known.

That is all a :class:`StateReader` does. ``step`` is cleanba's ``get_action``
with the per-cell state of *that pass* returned alongside the action, so the
same call that decides what the agent does also says what it was representing
when it decided. One reader per architecture, chosen from the policy's
``PolicySpec`` by :func:`state_reader_for`, and nothing outside this module
needs to know which network it is talking to.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Callable

import jax
import numpy as np
from cleanba.convlstm import ConvLSTMConfig
from cleanba.network import GuezConvSequence, GuezResNetConfig, Policy, PolicySpec

from goalmisgen.nets.scaled import ScaledInputSpec
from goalmisgen.nets.transformer import TransformerSpec, residual_grids


@dataclasses.dataclass(frozen=True)
class PerCellState:
    """What one forward pass held at every cell, layer by layer.

    Each entry is ``(batch, height, width, channels)``. ``features`` is the
    state the rest of the network can read — the DRC's ``h``, a ResNet stage's
    output, a transformer block's residual stream. ``cell_state`` is the DRC's
    ``c``, the memory the recurrence carries and the site the steering
    experiments write to; networks with nothing carried between steps have no
    such thing and leave it ``None``.
    """

    features: tuple[np.ndarray, ...]
    cell_state: tuple[np.ndarray, ...] | None = None

    def stacked(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """One environment's grids, layers concatenated on the channel axis.

        Returns ``(features, cell_state)`` each ``(height, width, n_layers *
        channels)`` — the layout :class:`~goalmisgen.analysis.activations.Rollout`
        stores and :func:`~goalmisgen.analysis.probes.layer_slice` divides. A
        network without a cell state gets its features back in that slot, so
        code that only reads shapes keeps working; code that *writes* to it
        should ask :attr:`StateReader.has_cell_state` first.
        """
        features = np.concatenate([np.asarray(g)[index] for g in self.features], axis=-1)
        if self.cell_state is None:
            return features, features
        cell_state = np.concatenate([np.asarray(g)[index] for g in self.cell_state], axis=-1)
        return features, cell_state


class StateReader:
    """``get_action`` that also returns the per-cell state it computed.

    Subclasses say which intermediates to capture and how to turn them into
    grids. ``layer_names`` label the grids in order; ``n_layers`` is what
    :func:`~goalmisgen.analysis.probes.layer_slice` needs.
    """

    layer_names: tuple[str, ...]
    has_cell_state: bool = False

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._step = jax.jit(
            partial(
                policy.apply,
                method=policy.get_action,
                mutable=["intermediates"],
                capture_intermediates=self._capture_filter(),
            ),
            static_argnames="temperature",
        )

    @property
    def n_layers(self) -> int:
        return len(self.layer_names)

    def step(self, params, carry, observations, episode_starts, key, temperature: float = 0.0):
        """``(carry, action, logits, key, state)`` for one batch of observations.

        The first four are exactly what ``policy.get_action`` returns for the
        same arguments — the reader adds the state, it does not change the
        decision — so a rollout can use this in place of ``get_action`` without
        its actions moving.
        """
        (carry, action, logits, key), variables = self._step(
            params, carry, observations, episode_starts, key, temperature=temperature
        )
        return carry, action, logits, key, self._state(carry, variables.get("intermediates", {}))

    def state_of_carry(self, carry) -> PerCellState:
        """The state held in a carry alone, with no forward pass.

        Only a recurrent network has one. Needed when the carry has been edited
        — steering writes into it — and the probe must read what was written
        rather than what the next pass would compute.
        """
        raise ValueError(f"{type(self).__name__} reads a network with no state between steps; there is no carry to read")

    # -- per-architecture -------------------------------------------------

    def _capture_filter(self) -> Callable | bool:
        return False

    def _state(self, carry, intermediates: dict) -> PerCellState:
        raise NotImplementedError


class DRCReader(StateReader):
    """The ConvLSTM's carry, layer by layer: ``h`` as features, ``c`` as cell state."""

    has_cell_state = True

    def __init__(self, policy: Policy) -> None:
        super().__init__(policy)
        self.layer_names = tuple(f"cell_list_{i}" for i in range(policy.cfg.n_recurrent))

    def _state(self, carry, intermediates: dict) -> PerCellState:
        return self.state_of_carry(carry)

    def state_of_carry(self, carry) -> PerCellState:
        return PerCellState(
            features=tuple(np.asarray(layer.h) for layer in carry),
            cell_state=tuple(np.asarray(layer.c) for layer in carry),
        )


class ResNetReader(StateReader):
    """Each ``GuezConvSequence``'s output — the feature map after every stage.

    Found by name anywhere under ``network_params``, so the ResNet may sit
    behind an input-scaling wrapper without the reader caring.
    """

    def __init__(self, policy: Policy) -> None:
        super().__init__(policy)
        self.layer_names = tuple(f"GuezConvSequence_{i}" for i in range(len(unwrap(policy.cfg).channels)))

    def _capture_filter(self):
        return lambda module, method_name: isinstance(module, GuezConvSequence) and method_name == "__call__"

    def _state(self, carry, intermediates: dict) -> PerCellState:
        found = _find_named(intermediates["network_params"], set(self.layer_names))
        missing = [name for name in self.layer_names if name not in found]
        if missing:
            raise RuntimeError(f"forward pass recorded no output for {missing}")
        return PerCellState(features=tuple(np.asarray(found[name]["__call__"][0]) for name in self.layer_names))


def unwrap(spec) -> PolicySpec:
    """The network spec inside any input-scaling wrapper."""
    while isinstance(spec, ScaledInputSpec):
        spec = spec.inner
    return spec


def _find_named(tree: dict, names: set[str]) -> dict:
    """Sub-dicts of a flax collection whose key is one of ``names``, at any depth."""
    out = {}
    for key, value in tree.items():
        if key in names:
            out[key] = value
        elif isinstance(value, dict):
            out.update(_find_named(value, names))
    return out


class TransformerReader(StateReader):
    """The residual stream as an (H, W, d_model) grid: after the embedding, then after each block."""

    def __init__(self, policy: Policy) -> None:
        super().__init__(policy)
        self.layer_names = ("embed",) + tuple(f"block_{i}" for i in range(unwrap(policy.cfg).n_layers))

    def _state(self, carry, intermediates: dict) -> PerCellState:
        grids = residual_grids(intermediates)
        if len(grids) != len(self.layer_names):
            raise RuntimeError(f"expected {len(self.layer_names)} residual grids, the pass recorded {len(grids)}")
        return PerCellState(features=tuple(np.asarray(g) for g in grids))


READERS: dict[type, type[StateReader]] = {
    ConvLSTMConfig: DRCReader,
    GuezResNetConfig: ResNetReader,
    TransformerSpec: TransformerReader,
}


def state_reader_for(policy: Policy) -> StateReader:
    """The reader for whatever network this policy wraps."""
    spec = unwrap(policy.cfg)
    for spec_type, reader_type in READERS.items():
        if isinstance(spec, spec_type):
            return reader_type(policy)
    raise TypeError(f"no state reader for {type(spec).__name__}; add one to goalmisgen.nets.readers.READERS")
