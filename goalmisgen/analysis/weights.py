"""Comparing what fine-tuning did to the weights, across a grid of tasks.

A single weight diff says very little. The gradient writes wherever it is
cheapest to write, so *where* a diff lands is not evidence about where a
quantity is represented, and its sparsity is a property of whatever penalty
produced it. What a single diff cannot be is *collinear with another one*.

So the object of interest here is a family of diffs indexed by a task parameter
— what an objective is worth, in the experiment this was written for. If the
network holds that parameter in something like a slot, moving it should move the
weights along one direction by an amount that tracks the change, and the family
collapses to a line. If instead each task was solved by rebuilding a threshold,
the diffs are unrelated directions and no line fits them.

The axis is deliberately scaled *per unit of the parameter* rather than
normalised: the magnitude is the part of the claim that a heading alone would
not test, and keeping it is what allows a value to be written by adding its
offset.
"""

from __future__ import annotations

import numpy as np


def fit_axis(offsets: np.ndarray, diffs: np.ndarray) -> np.ndarray:
    """Least-squares ``diff = offset * axis`` through the origin.

    ``offsets`` are signed distances from whatever the base model was trained at,
    so the fit is forced through zero: a zero change must mean a zero diff, and
    an intercept would otherwise absorb the drift that the null arm exists to
    measure separately.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    diffs = np.asarray(diffs, dtype=np.float64)
    if offsets.ndim != 1 or diffs.ndim != 2 or len(offsets) != len(diffs):
        raise ValueError(f"need one offset per diff, got {offsets.shape} and {diffs.shape}")
    denominator = float(offsets @ offsets)
    if denominator < 1e-12:
        raise ValueError("every arm sits at the base value, so there is no offset to fit against")
    return (offsets @ diffs) / denominator


def explained(diff: np.ndarray, offset: float, axis: np.ndarray) -> float:
    """Fraction of one diff's squared length that ``offset * axis`` accounts for.

    Can go negative, and should be allowed to: a diff pointing away from the axis
    is worse than no prediction at all, and clipping that at zero would hide the
    one outcome that refutes the hypothesis.
    """
    diff = np.asarray(diff, dtype=np.float64)
    total = float(np.sum(diff**2))
    if total < 1e-30:
        raise ValueError("this arm did not move, so there is nothing to explain")
    return 1 - float(np.sum((diff - offset * np.asarray(axis, dtype=np.float64)) ** 2)) / total


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two diffs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-30:
        raise ValueError("a diff of zero length has no direction")
    return float(a @ b) / denominator


def projected_offset(diff: np.ndarray, axis: np.ndarray) -> float:
    """How far along ``axis`` a diff reaches, in units of the axis's parameter.

    Read against the offset the arm was actually trained at, this is the axis
    tested as a *scale* rather than as a heading: an axis fitted without this arm
    should recover not merely its direction but how far it went.
    """
    axis = np.asarray(axis, dtype=np.float64)
    denominator = float(axis @ axis)
    if denominator < 1e-30:
        raise ValueError("a zero axis has no scale")
    return float(np.asarray(diff, dtype=np.float64) @ axis) / denominator
