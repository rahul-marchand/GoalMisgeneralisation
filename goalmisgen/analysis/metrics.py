"""Scoring, for both probe families. Pure array statistics.

Nothing here knows what a maze is. It takes ``(n,)`` arrays of labels,
predictions, strata and episode ids, and that is the whole interface — which is
also the boundary a port to another environment would follow, so
``tests/test_metrics.py`` asserts this module imports nothing from the
environment layer.

Two ideas do most of the work.

**Correlation and R² answer different questions, and the difference is
recoverable exactly.** A probe can order every cell correctly and still score
badly under R² if its predictions are systematically compressed — which is what
a shrinking regulariser produces. :func:`calibration` decomposes R² into an
ordering term and two calibration terms, so "the network's magnitudes are wrong"
can be told apart from "the fit shrank them".

**Comparisons must be made within groups the confound cannot separate.**
:func:`stratified_correlation` and :func:`stratified_auc` are the same idea in
regression and classification form: the matched negatives that rescued the
distance-band result were stratification, and generalising them means a target
declaring its confound gets the defence automatically.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    """Variance explained, against the mean of *these* rows.

    Scoring a subset against a wider population's mean would credit a probe for
    the subset simply having a larger target than average.
    """
    if len(y) < 2:
        return float("nan")
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - prediction) ** 2)) / total


@dataclasses.dataclass(frozen=True)
class Calibration:
    """R² split into how well a probe orders, and how badly it is scaled.

    ``r2 == shape_r2 - scale_loss - offset_loss`` exactly, from three moments.
    """

    r2: float
    shape_r2: float
    """The R² an oracle affine rescaling would reach — the ordering ceiling."""

    scale_loss: float
    """Spread wrong for the correlation achieved. Positive under shrinkage."""

    offset_loss: float
    """Level wrong."""

    slope: float
    """Least-squares slope of the label on the prediction. 1.0 is calibrated;
    above 1 means the probe compresses and must be stretched to fit."""

    bias: float
    """Mean prediction minus mean label, in the label's own units."""


def calibration(y: np.ndarray, prediction: np.ndarray) -> Calibration:
    """Decompose R² into ordering and calibration.

    With rho = corr(p, y), k = sd(p)/sd(y) and b = (mean(p) - mean(y))/sd(y),

        R² = rho² - (rho - k)² - b²

    which is exact, not an approximation. ``shape_r2`` is invariant to any
    affine map of the prediction, so it is the calibration-immune statistic.
    """
    if len(y) < 2:
        return Calibration(*[float("nan")] * 6)

    spread = float(y.std())
    predicted_spread = float(prediction.std())
    if spread <= 0:
        return Calibration(*[float("nan")] * 6)

    rho = 0.0 if predicted_spread <= 0 else float(np.corrcoef(prediction, y)[0, 1])
    k = predicted_spread / spread
    b = (float(prediction.mean()) - float(y.mean())) / spread

    return Calibration(
        r2=rho**2 - (rho - k) ** 2 - b**2,
        shape_r2=rho**2,
        scale_loss=(rho - k) ** 2,
        offset_loss=b**2,
        slope=float("nan") if k <= 0 else rho / k,
        bias=float(prediction.mean()) - float(y.mean()),
    )


def affine_fit(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Scale and offset mapping ``prediction`` onto ``y``, by least squares.

    Fitted on the training split and applied to the test split, so it is a
    correction to the fit rather than a peek at the answer. It removes the
    shrinkage a penalised regression introduces without touching the ordering,
    which is what the probe is actually being asked about.
    """
    variance = float(prediction.var())
    if variance <= 0:
        return 0.0, float(y.mean())
    scale = float(np.cov(prediction, y, bias=True)[0, 1]) / variance
    return scale, float(y.mean()) - scale * float(prediction.mean())


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC; ties share their average rank."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _centre_within(values: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each stratum's mean; report which rows had anything to compare."""
    out = values.astype(np.float64).copy()
    usable = np.zeros(len(out), dtype=bool)
    for value in np.unique(stratum):
        rows = stratum == value
        if rows.sum() < 2:
            continue
        out[rows] -= out[rows].mean()
        usable |= rows
    return out, usable


def stratified_correlation(prediction: np.ndarray, y: np.ndarray, stratum: np.ndarray) -> float:
    """Correlation after centring both sides within each stratum.

    Whatever the stratum determines is removed by construction, so a feature
    encoding nothing but the confound scores zero. Strata with one member are
    dropped rather than centred to nothing: a lone point carries no
    within-stratum information and keeping it only dilutes the estimate.

    Zero residual variance returns 0.0 — the confound accounted for everything,
    which is a result. ``NaN`` means there were no comparable rows at all.
    """
    p, usable = _centre_within(prediction, stratum)
    t, _ = _centre_within(y, stratum)
    if usable.sum() < 2:
        return float("nan")
    if p[usable].std() < 1e-12 or t[usable].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p[usable], t[usable])[0, 1])


def stratified_auc(y: np.ndarray, scores: np.ndarray, stratum: np.ndarray) -> float:
    """AUC over pairs the stratum cannot separate, pooled across strata.

    The classification twin of :func:`stratified_correlation`, and the
    generalisation of matched negatives: comparing a positive only against
    negatives in its own stratum is exactly restricting to comparable pairs.
    Weighted by pair count, so large strata count for more.
    """
    total_pairs = 0.0
    weighted = 0.0
    for value in np.unique(stratum):
        rows = stratum == value
        labels = y[rows]
        n_pos, n_neg = float(labels.sum()), float(len(labels) - labels.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        pairs = n_pos * n_neg
        weighted += pairs * roc_auc(labels, scores[rows])
        total_pairs += pairs
    return weighted / total_pairs if total_pairs else float("nan")


def _resample(statistic, episode: np.ndarray, resamples: int, seed: int):
    """Yield the statistic over ``resamples`` whole-episode resamples."""
    unique = np.unique(episode)
    rows_of = {value: np.flatnonzero(episode == value) for value in unique}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        yield statistic(np.concatenate([rows_of[unique[i]] for i in chosen]))


def bootstrap_episodes(statistic, episode: np.ndarray, resamples: int = 200, seed: int = 0, level: float = 0.95):
    """Resample whole episodes, never cells.

    Cells within an episode share one maze and one label field, so treating them
    as independent understates the uncertainty by roughly 1.6x and reports a far
    larger experiment than this is.
    """
    values = [value for value in _resample(statistic, episode, resamples, seed) if np.isfinite(value)]
    if not values:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def select_rows(episode: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Row indices belonging to ``chosen`` episode ids, repeats included."""
    rows_of = {value: np.flatnonzero(episode == value) for value in np.unique(episode)}
    return np.concatenate([rows_of[value] for value in chosen if value in rows_of])


def bootstrap_paired(
    statistic_a, statistic_b, episodes: np.ndarray, resamples: int = 200, seed: int = 0, level: float = 0.95
):
    """Interval for ``a - b`` over a common resample of *episodes*.

    Two overlapping confidence intervals do not mean two quantities are equal.
    Sharing the resample cancels the between-episode variance both statistics
    carry, so the interval on the difference is far tighter than either alone —
    which is what makes a comparison decisive rather than suggestive.

    Each statistic maps an array of episode ids to a float and looks up its own
    rows, so the two may be measured over different cell sets. They are whenever
    two targets mask differently, which is exactly the reached-versus-unreached
    case this exists for.
    """
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(resamples):
        chosen = unique[rng.integers(0, len(unique), len(unique))]
        difference = statistic_a(chosen) - statistic_b(chosen)
        if np.isfinite(difference):
            values.append(difference)

    if not values:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)
