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


def fit_axis_and_drift(offsets: np.ndarray, diffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares ``diff = drift + offset * axis``.

    Every arm carries a component that has nothing to do with the parameter it
    was trained on: the same number of updates on the same task moves the weights
    whether or not there is anything to learn, which is what the null arm
    measures directly. That component sits at every offset, including zero.

    Forcing the fit through the origin therefore does not remove it — it leaks it
    into the axis, in proportion to how far the offsets are from being balanced
    around zero. The symptom is an axis that reads back roughly the same offset
    for every arm, the null one included. The intercept absorbs it instead, so
    the axis is estimated from how the arms differ from *each other* rather than
    from how far they all moved from the base.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    diffs = np.asarray(diffs, dtype=np.float64)
    if offsets.ndim != 1 or diffs.ndim != 2 or len(offsets) != len(diffs):
        raise ValueError(f"need one offset per diff, got {offsets.shape} and {diffs.shape}")
    if len(offsets) < 3:
        raise ValueError("an intercept and a slope need at least three arms to be identified")
    centred = offsets - offsets.mean()
    denominator = float(centred @ centred)
    if denominator < 1e-12:
        raise ValueError("every arm sits at the same offset, so there is no slope to fit")
    axis = (centred @ (diffs - diffs.mean(axis=0))) / denominator
    return axis, diffs.mean(axis=0) - offsets.mean() * axis


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


def permutation_cosines(
    offsets: np.ndarray,
    diffs: np.ndarray,
    reference: np.ndarray,
    resamples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Cosines against ``reference`` when the offsets are shuffled among the diffs.

    The null this builds is *not* "the cosine is zero". Every arm's diff contains
    a large common component — the cost of running the updates, which the null
    arm measures and which is the same whatever value was trained — so two axes
    fitted from two sweeps of the same agent share structure whether or not
    anything about value is in there. Assuming a null of zero ignores that and
    reads the shared drift as evidence.

    Shuffling which offset belongs to which diff destroys the association between
    the parameter and the direction while leaving that shared structure exactly
    where it was. What comes back is the distribution of cosines obtainable from
    these diffs with no value signal in them, which is the thing an observed
    cosine has to beat.

    This replaces dividing by split-half reliability as the load-bearing
    statistic. That correction reached ×7 on the first grid, and in
    ``results/three-objective.txt`` it returned cosines outside the range a
    cosine can take — a correction announcing that it has broken down. A
    permutation null assumes nothing about how noise scales.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    diffs = np.asarray(diffs, dtype=np.float64)
    if offsets.ndim != 1 or diffs.ndim != 2 or len(offsets) != len(diffs):
        raise ValueError(f"need one offset per diff, got {offsets.shape} and {diffs.shape}")
    # Permuting the offsets does not touch the diffs, so everything that depends
    # on them is computed once. Writing the fitted axis as
    #
    #     axis = (c @ C) / (c @ c),   c = offsets - mean,  C = diffs - mean
    #
    # its cosine against the reference needs only ``C @ reference`` (one vector
    # per arm) and the Gram matrix ``C @ C.T`` (arms by arms). Both are tiny, and
    # each resample then costs a couple of dot products over the arms instead of
    # a fresh least squares over the whole parameter vector.
    #
    # This is not a micro-optimisation. Naively, one resample allocates and
    # traverses an arms-by-parameters matrix -- 400 MB for a 25-arm sweep of this
    # network -- and two thousand of them took longer than the rest of the
    # analysis put together, per sweep.
    centred_diffs = diffs - diffs.mean(axis=0)
    gram = centred_diffs @ centred_diffs.T
    projected = centred_diffs @ np.asarray(reference, dtype=np.float64)
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm < 1e-30:
        raise ValueError("a diff of zero length has no direction")

    rng = np.random.default_rng(seed)
    drawn = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        shuffled = rng.permutation(offsets)
        centred = shuffled - shuffled.mean()
        denominator = float(centred @ centred)
        if denominator < 1e-12:
            raise ValueError("every arm sits at the same offset, so there is no slope to fit")
        axis_norm = float(np.sqrt(max(centred @ gram @ centred, 0.0))) / denominator
        if axis_norm < 1e-30:
            # This permutation's offsets are orthogonal to every direction the
            # diffs vary in, so it fits no axis at all and says nothing about
            # alignment with the reference. Zero is the honest contribution.
            # It needs stating because it is reachable: a perfectly collinear
            # family has one direction to be orthogonal to. Real diffs never are,
            # and the earlier implementation hit the same case and returned
            # floating-point noise in the last few digits instead.
            drawn[index] = 0.0
            continue
        drawn[index] = float(centred @ projected) / denominator / (axis_norm * reference_norm)
    return drawn


def permutation_norms(
    offsets: np.ndarray,
    diffs: np.ndarray,
    resamples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """``|axis|`` when the offsets are shuffled among the diffs.

    The companion of :func:`permutation_cosines`, and the one that answers
    whether a rung has an axis *at all* rather than whether two axes agree.

    Comparing ``|axis|`` against zero is meaningless: least squares through a
    cloud of noisy diffs returns a nonzero slope whatever the diffs are, and how
    large depends on how noisy they are and on the grid's leverage, neither of
    which is constant across the things being compared. Shuffling which offset
    belongs to which diff destroys any association with value while leaving the
    diffs, their spread and the design exactly as they were, so what comes back
    is the length a slope of *this* grid over *these* diffs reaches with no value
    signal in it. That is the number ``|axis|`` has to beat.

    It also removes the need to read ``|axis|`` against ``|drift|``, which is a
    ratio of two things that scale differently with arm length and says nothing
    calibrated about either.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    diffs = np.asarray(diffs, dtype=np.float64)
    if offsets.ndim != 1 or diffs.ndim != 2 or len(offsets) != len(diffs):
        raise ValueError(f"need one offset per diff, got {offsets.shape} and {diffs.shape}")
    centred_diffs = diffs - diffs.mean(axis=0)
    gram = centred_diffs @ centred_diffs.T
    rng = np.random.default_rng(seed)
    drawn = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        shuffled = rng.permutation(offsets)
        centred = shuffled - shuffled.mean()
        denominator = float(centred @ centred)
        if denominator < 1e-12:
            raise ValueError("every arm sits at the same offset, so there is no slope to fit")
        drawn[index] = float(np.sqrt(max(centred @ gram @ centred, 0.0))) / denominator
    return drawn


def split_half_reliability(
    offsets: np.ndarray,
    diffs: np.ndarray,
    splits: int = 200,
    seed: int = 0,
) -> float:
    """How much of a fitted axis is signal rather than fine-tuning noise.

    The grid is mirrored, so it is made of ``(+m, -m)`` pairs. Splitting *pairs*
    rather than arms keeps each half balanced about the base, which matters: an
    unbalanced half lets the common fine-tuning component leak into its axis, and
    two halves that both leaked it would agree about the leak and report it as
    reliability. Each half then gives an independent estimate of the same
    direction, and the cosine between them is what the estimate is worth.

    Averaged over random splits and corrected to full length by Spearman-Brown,
    ``2r / (1 + r)``, because each half is fitted on half the arms and a half-
    length reliability understates the whole.

    ``016`` does this with the two widest pairs, which is all a seven-point grid
    affords. On a twelve-pair grid that throws away five sixths of the splits
    available and gives one noisy number instead of a distribution.

    Returns ``nan`` when there are fewer than four pairs, which cannot be split
    into two halves that each fit a slope.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    diffs = np.asarray(diffs, dtype=np.float64)
    table = {round(float(o), 6): index for index, o in enumerate(offsets)}
    pairs = [
        (table[m], table[-m])
        for m in sorted({abs(float(o)) for o in offsets if abs(o) > 1e-9}, reverse=True)
        if m in table and -m in table
    ]
    if len(pairs) < 4:
        return float("nan")

    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(splits):
        order = rng.permutation(len(pairs))
        half = len(pairs) // 2
        estimates = []
        for chosen in (order[:half], order[half : 2 * half]):
            rows = [index for pair in (pairs[i] for i in chosen) for index in pair]
            axis, _ = fit_axis_and_drift(offsets[rows], diffs[rows])
            estimates.append(axis)
        if min(np.linalg.norm(e) for e in estimates) < 1e-30:
            continue
        cosines.append(cosine(estimates[0], estimates[1]))
    if not cosines:
        return float("nan")
    half_length = float(np.mean(cosines))
    if half_length <= -1.0:
        return float("nan")
    return float(2 * half_length / (1 + half_length))


def permutation_p_value(observed: float, null: np.ndarray, alternative: str = "less") -> float:
    """How often the null reaches at least as far as the observation.

    ``less`` for the one-knob prediction, where the interesting direction is
    toward −1; ``greater`` for the hierarchical three-objective prediction, whose
    whole content is that the cosine is *positive*. The +1 in numerator and
    denominator is the standard correction that stops a p-value of exactly zero
    being reported from a finite number of resamples.
    """
    null = np.asarray(null, dtype=np.float64)
    if alternative == "less":
        extreme = int(np.sum(null <= observed))
    elif alternative == "greater":
        extreme = int(np.sum(null >= observed))
    elif alternative == "two-sided":
        extreme = int(np.sum(np.abs(null) >= abs(observed)))
    else:
        raise ValueError(f"alternative should be 'less', 'greater' or 'two-sided', got {alternative!r}")
    return (extreme + 1) / (len(null) + 1)


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
