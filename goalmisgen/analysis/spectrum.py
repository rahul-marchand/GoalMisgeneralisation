"""How many directions a family of fine-tuning diffs actually spans.

:mod:`goalmisgen.analysis.weights` asks whether the family collapses to a line
and how well that line predicts a held-out arm. It cannot say what is left over.
A family can score well on both while carrying a second, perfectly real
direction alongside the first — the leave-one-out fit only reports that the
axis explains a lot, not that nothing else replicates.

That distinction is the whole content of the width/depth campaign. The task has
**one goal degree of freedom**: ``015``/``028`` established that the knob is the
gap between two objective values rather than the values themselves, so the
correct rank is one and *any* second direction that replicates is the network's
doing rather than the task's. Very few settings hand you the true rank; this one
does, so it is worth measuring against.

Two numbers come out.

``participation_ratio``   the effective number of directions in the family once
                          the common fine-tuning component is removed. One for a
                          perfect line, larger as the family fans out. Purely
                          descriptive: fine-tuning noise inflates it, and how
                          much depends on the arm count and the network, neither
                          of which is constant across a grid of shapes.
``residual_reliability``  the inferential one. Fit the axis on half the arms,
                          take the leading direction of *what the axis did not
                          explain*, do the same on the other half, and ask
                          whether the two agree. A second axis that replicates
                          across disjoint halves is structure; one that does not
                          is the shape of this fine-tune's noise.

Everything here is computed from the arms-by-arms Gram matrix and the offsets,
never from the diffs themselves. That is not an optimisation detail. The drift,
the axis and hence every residual are *linear combinations of the arms*, so the
whole calculation lives in a space of dimension ``n_arms``; going through the
diffs would mean materialising an arms-by-parameters matrix, which for a 50M
parameter model and 24 arms is 9.6 GB in float64 and is the reason the naive
version of this module did not run at the top of the grid.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.analysis.weights import gram_matrix, mirrored_pairs

__all__ = [
    "axis_removed_operator",
    "drift_removed_operator",
    "gram_matrix",
    "participation_ratio",
    "permutation_participation_ratio",
    "residual_reliability",
    "spectrum",
    "variance_shares",
]


def drift_removed_operator(offsets: np.ndarray) -> np.ndarray:
    """``W`` with ``W @ diffs`` equal to each arm minus the fitted drift.

    The drift is the intercept of ``diff = drift + offset * axis`` — the part
    every arm carries because it was fine-tuned at all, which the null arm
    measures directly. Removing it is what leaves the arms comparable to each
    other rather than to the base.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 1:
        raise ValueError(f"offsets must be one-dimensional, got shape {offsets.shape}")
    n = len(offsets)
    if n < 3:
        raise ValueError("an intercept and a slope need at least three arms to be identified")
    centred = offsets - offsets.mean()
    denominator = float(centred @ centred)
    if denominator < 1e-12:
        raise ValueError("every arm sits at the same offset, so there is no slope to fit")
    axis_weights = centred / denominator
    drift_weights = np.full(n, 1.0 / n) - offsets.mean() * axis_weights
    return np.eye(n) - drift_weights[None, :]


def axis_removed_operator(offsets: np.ndarray) -> np.ndarray:
    """``W`` with ``W @ diffs`` equal to each arm's residual about the fitted line.

    Drift *and* ``offset * axis`` removed, so what is left is whatever the
    one-knob account does not explain. This is the family whose leading
    direction :func:`residual_reliability` tests for replication.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    centred = offsets - offsets.mean()
    denominator = float(centred @ centred)
    return drift_removed_operator(offsets) - np.outer(offsets, centred / denominator)


def spectrum(gram: np.ndarray) -> np.ndarray:
    """Singular values of whatever family produced ``gram``, descending.

    ``gram`` is ``R @ R.T`` for some arms-by-parameters ``R``; its eigenvalues
    are the squared singular values of ``R``. Tiny negative eigenvalues come
    back from any finite-precision symmetric eigensolver on a rank-deficient
    matrix — the operators above are deliberately rank-deficient, by two — and
    are clipped rather than reported as complex numbers.
    """
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"expected a square Gram matrix, got shape {gram.shape}")
    eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2)
    return np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1]


def variance_shares(singular_values: np.ndarray) -> np.ndarray:
    """Fraction of squared length carried by each direction, descending."""
    squares = np.asarray(singular_values, dtype=np.float64) ** 2
    total = float(squares.sum())
    if total < 1e-30:
        raise ValueError("this family has no length, so its variance does not divide")
    return squares / total


def participation_ratio(singular_values: np.ndarray) -> float:
    """Effective number of directions: ``(sum s^2)^2 / sum s^4``.

    One when a single direction carries everything, ``k`` when ``k`` directions
    carry equal shares. Preferred to counting singular values above a threshold
    because no threshold is comparable across networks of different sizes.
    """
    squares = np.asarray(singular_values, dtype=np.float64) ** 2
    denominator = float((squares**2).sum())
    if denominator < 1e-30:
        raise ValueError("this family has no length, so it has no participation ratio")
    return float(squares.sum() ** 2 / denominator)


def _leading(gram: np.ndarray) -> tuple[np.ndarray, float]:
    """Leading eigenvector of a residual Gram, and the singular value with it."""
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
    return eigenvectors[:, -1], float(np.sqrt(max(eigenvalues[-1], 0.0)))


def residual_reliability(
    offsets: np.ndarray,
    gram: np.ndarray,
    splits: int = 200,
    seed: int = 0,
) -> float:
    """Does the leading *residual* direction replicate across disjoint arms?

    Each split takes the mirrored ``(+m, -m)`` pairs, deals them into two halves,
    and within each half independently fits the axis, forms the residuals and
    extracts their leading direction. The two directions are then compared.
    Pairs rather than arms, for the reason in
    :func:`goalmisgen.analysis.weights.split_half_reliability`: an unbalanced
    half leaks the common fine-tuning component into its own axis, and two halves
    that both leaked it would agree about the leak.

    **Compared by absolute cosine**, because a singular vector is defined only up
    to sign and the two halves have no shared convention to fix it. In a space
    of a million parameters the absolute cosine of two unrelated directions is
    of order ``sqrt(2 / pi P)``, which is not distinguishable from zero here, so
    nothing is bought by the sign and a spurious anti-alignment is not counted as
    agreement.

    Spearman-Brown corrected to full length, as the axis reliability is. Note
    which way that cuts: the correction *inflates*, and the campaign's registered
    prediction is that this number stays **low**. An inflated estimate therefore
    makes that prediction harder to keep, which is the safe direction.

    Returns ``nan`` when there are fewer than four mirrored pairs, which cannot
    be split into two halves that each support a fit.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    if len(offsets) != len(gram):
        raise ValueError(f"{len(offsets)} offsets against a {len(gram)}-arm Gram matrix")
    pairs = mirrored_pairs(offsets)
    if len(pairs) < 4:
        return float("nan")

    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(splits):
        order = rng.permutation(len(pairs))
        half = len(pairs) // 2
        rows = [
            [index for pair in (pairs[i] for i in chosen) for index in pair]
            for chosen in (order[:half], order[half : 2 * half])
        ]
        operators, vectors, scales = [], [], []
        for chosen in rows:
            operator = axis_removed_operator(offsets[chosen])
            direction, scale = _leading(operator @ gram[np.ix_(chosen, chosen)] @ operator.T)
            operators.append(operator)
            vectors.append(direction)
            scales.append(scale)
        if min(scales) < 1e-30:
            # One half's arms sit exactly on their own fitted line, so it has no
            # residual direction to agree about. Reachable with few arms, and
            # skipping is honest where contributing zero would not be.
            continue
        cross = operators[0] @ gram[np.ix_(rows[0], rows[1])] @ operators[1].T
        cosines.append(abs(float(vectors[0] @ cross @ vectors[1]) / (scales[0] * scales[1])))
    if not cosines:
        return float("nan")
    half_length = float(np.mean(cosines))
    return float(2 * half_length / (1 + half_length))


def permutation_participation_ratio(
    offsets: np.ndarray,
    gram: np.ndarray,
    resamples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Participation ratio of the residuals when offsets are shuffled among arms.

    The companion null to the ones in :mod:`goalmisgen.analysis.weights`, and
    needed for the same reason: an observed participation ratio of 2.4 means
    nothing on its own, because removing a line fitted to *noise* leaves a
    different amount of structure than removing one fitted to signal, and how
    much depends on the arm count and the network. Shuffling which offset belongs
    to which diff destroys the association with value while leaving the diffs and
    the design where they were, so what comes back is the ratio this family
    reaches with no value signal in it.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    if len(offsets) != len(gram):
        raise ValueError(f"{len(offsets)} offsets against a {len(gram)}-arm Gram matrix")
    rng = np.random.default_rng(seed)
    drawn = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        operator = axis_removed_operator(rng.permutation(offsets))
        drawn[index] = participation_ratio(spectrum(operator @ gram @ operator.T))
    return drawn
