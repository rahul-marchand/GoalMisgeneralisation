"""Adding a direction to the recurrent state, calibrated in the task's units.

A probe gives weights ``w`` such that ``w . x`` decodes a quantity — for the
distance field, in cells. That same fit hands back a *direction*: the smallest
change to the activations that moves the decoded value by a chosen amount. Add
it during a rollout and the question stops being "is this information present"
and becomes "is it used".

The calibration is the point. Shifting the decoded distance to an objective by
``alpha`` cells predicts a shift of ``alpha`` in the distance at which the agent
abandons it — a number, not a direction of change. A slope of one says the field
*is* the compared quantity; a slope near zero says it is decodable but ignored.

Two details decide whether the arithmetic is right, and both are easy to get
wrong silently:

The probe is fitted on **standardised** inputs, so a unit of ``w`` is not a unit
of activation. The effective weight in raw space is ``w / std``, and the
minimum-norm shift achieving ``alpha`` is ``alpha * u / |u|^2`` for that ``u``.

The probe reads the concatenated hidden states of every layer, so a direction
must be **split back across layers** in the order they were concatenated.
:func:`verify` re-decodes a steered activation and checks the shift is what was
asked for, because an off-by-one in that split produces a plausible number
rather than a crash.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Direction:
    """A calibrated steering direction over the concatenated hidden state."""

    name: str
    delta: np.ndarray
    """(depth,) the activation change that moves the decoded value by +1."""

    def scaled(self, alpha: float) -> np.ndarray:
        return alpha * self.delta

    @property
    def unit_norm(self) -> float:
        """Size of the activation change one unit of the decoded quantity costs."""
        return float(np.linalg.norm(self.delta))


def from_probe(name: str, weights: np.ndarray, std: np.ndarray) -> Direction:
    """The minimum-norm activation change that moves this probe's output by one.

    ``weights`` is what ``fit_ridge`` returned, whose final entry is the bias and
    plays no part in a direction.
    """
    effective = weights[:-1] / std
    norm = float(effective @ effective)
    if norm <= 0:
        raise ValueError(f"probe {name!r} has no usable direction; its weights are all zero")
    return Direction(name, effective / norm)


def matched_random(name: str, reference: Direction, seed: int = 0) -> Direction:
    """A random direction of the same size as ``reference``.

    The control without which a steering result means nothing: a large enough
    perturbation disturbs any network, so "we added a vector and behaviour
    changed" is only informative against a vector that should do nothing.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=reference.delta.shape)
    return Direction(name, raw / np.linalg.norm(raw) * reference.unit_norm)


def matched(name: str, other: Direction, reference: Direction) -> Direction:
    """``other``'s direction rescaled to ``reference``'s magnitude.

    So a control differs from the real direction in *where it points* and
    nothing else.
    """
    return Direction(name, other.delta / other.unit_norm * reference.unit_norm)


def apply_to_carry(carry, delta: np.ndarray):
    """Add ``delta`` to every cell of every layer's hidden state.

    Split across layers in concatenation order, which is the order the probe
    read them in. A constant added at every cell shifts the whole decoded field
    uniformly — "this objective is alpha cells further from everywhere" — which
    is the intervention the calibration describes.
    """
    steered, start = [], 0
    for layer in carry:
        width = layer.h.shape[-1]
        if start + width > len(delta):
            raise ValueError(f"direction has {len(delta)} entries but the carry needs at least {start + width}")
        steered.append(layer.replace(h=layer.h + delta[start : start + width]))
        start += width
    if start != len(delta):
        raise ValueError(f"direction has {len(delta)} entries but the carry consumed {start}")
    return steered


def verify(direction: Direction, weights: np.ndarray, mean: np.ndarray, std: np.ndarray, alpha: float) -> float:
    """Decoded shift actually produced by steering ``alpha``. Should equal ``alpha``.

    Cheap, and it catches the two silent failures: standardising in the wrong
    space, and splitting the direction across layers in the wrong order.
    """
    from goalmisgen.analysis.probes import apply_linear

    baseline = np.zeros((1, len(direction.delta)))
    before = apply_linear(baseline, weights, mean, std)
    after = apply_linear(baseline + direction.scaled(alpha), weights, mean, std)
    return float(after[0] - before[0])


def from_contrast(name: str, high: np.ndarray, low: np.ndarray, weights: np.ndarray, std: np.ndarray) -> Direction:
    """Mean activation where the quantity is large, minus where it is small.

    This is what the steering literature actually uses, and it differs from a
    probe's weight vector in a way that matters. Ridge finds the *minimum-norm*
    direction that predicts, which is free to point almost anywhere the network
    does not vary — it can decode perfectly while aiming somewhere the network
    never goes. A difference of means lies in the data manifold by construction:
    it is a direction the network is observed to travel along when the quantity
    changes.

    The probe is still used, but only to calibrate: the returned direction is
    scaled so one unit of it moves the decoded value by one.
    """
    raw = np.asarray(high).mean(axis=0) - np.asarray(low).mean(axis=0)
    per_unit = float((weights[:-1] / std) @ raw)
    if abs(per_unit) < 1e-12:
        raise ValueError(
            f"the contrast for {name!r} moves the decoded value by {per_unit:.2e} per unit, so it cannot be "
            "calibrated; the two groups are not separated along the probed quantity"
        )
    return Direction(name, raw / per_unit)


def write_to_cell_state(carry, deltas):
    """Add a per-cell grid to each layer's ConvLSTM **cell state**.

    ``deltas`` is one array per layer, shaped ``(n_envs, height, width,
    channels)`` — a spatial edit, not a constant. That is the difference between
    this and :func:`apply_to_carry`, and it is the whole reason this exists.

    :func:`apply_to_carry` adds the same vector at every cell, which for a field
    is a uniform offset: the argmin and the gradient are unchanged, so a policy
    reading the field's *shape* is provably unaffected and the null it produces
    is guaranteed in advance. Writing a plan means writing different things at
    different cells.

    The cell state rather than the hidden state, for the same causal reason: the
    hidden state is recomputed from the gates on the next tick, so an edit to it
    is erased by the next forward pass, while ``c`` is the layer's persistent
    memory and an edit there is something the recurrence has to carry.
    """
    if len(deltas) != len(carry):
        raise ValueError(f"{len(deltas)} deltas for {len(carry)} layers")
    steered = []
    for layer, delta in zip(carry, deltas):
        if delta.shape != layer.c.shape:
            raise ValueError(f"delta shaped {delta.shape} cannot be added to a cell state shaped {layer.c.shape}")
        steered.append(layer.replace(c=layer.c + delta))
    return steered
