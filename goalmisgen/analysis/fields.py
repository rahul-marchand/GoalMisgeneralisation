"""Fitting a per-cell probe and scoring it against its target's own null.

The route probe asks a yes/no question and is scored with AUC. This asks a
magnitude — how many steps is this cell from the objective — which needs a
regression fit and a different defence.

The defence is against **the target's declared confound**. For a distance field
that is straight-line geometry: on real levels ``corr(bfs, manhattan) = 0.57``
and a linear model on it alone explains R² = 0.33 of the true field, which is
enough to look like a finding. Pooled R² is the regression analogue of the
pooled negatives that invalidated the first distance-band result, so it is
printed for context and never rested on.

The claim rests on two numbers:

**Hard-cell R²** — accuracy restricted to cells where the confound is most
wrong (``label - confound >= tau``), scored against that subset's own mean
because the subset has a systematically larger target and the global mean would
flatter every arm.

**Stratified partial correlation** — the exact generalisation of matched
negatives: centre prediction and label within each integer confound value, then
correlate the residuals.

Nothing here knows what a maze is; that lives in
:mod:`goalmisgen.analysis.targets`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from goalmisgen.analysis import metrics
from goalmisgen.analysis.probes import Feature, apply_linear, fit_ridge

DEGENERATE = 1e-6
"""A channel whose spread is below this carries no information.

Standardising divides by ``std + 1e-8``, so a near-constant channel is
amplified into unit-scale noise that the penalty must then suppress. Left in,
they weaken exactly the arms with the least signal — which are the controls.
"""


@dataclasses.dataclass(frozen=True)
class CellData:
    """Per-cell rows, masked identically across arms so they are comparable."""

    x: np.ndarray
    """(n, depth) features."""

    y: np.ndarray
    """(n,) the target's label, in its own units."""

    confound: np.ndarray
    """(n, k) the target's declared null. Column 0 lower-bounds ``y``."""

    episode: np.ndarray
    """(n,) which rollout each row came from, for grouped resampling."""

    mask_fraction: float
    """Share of cells that survived masking."""

    dropped_columns: int
    """Feature channels removed for carrying no variation."""


@dataclasses.dataclass(frozen=True)
class FieldResult:
    name: str
    hard_r2: float
    """The headline: R² on cells the confound gets most wrong, recalibrated."""

    hard_interval: tuple[float, float]
    hard_shape_r2: float
    """The ordering ceiling on those cells — R² an oracle rescaling would reach."""

    pooled_r2: float
    partial_r: float
    partial_r_within_episode: float
    """The same, stratified within episode as well as by confound. If this
    collapses, the pooled figure was riding on between-maze variance."""

    slope: float
    bias: float
    mae: float
    n_samples: int
    n_hard: int
    depth: int
    dropped_columns: int
    l2: float
    mask_fraction: float
    feature_norm: float
    sensitivity: tuple[tuple[float, float], ...]


def cell_data(rollouts, feature: Feature, target, drop_degenerate: bool = True) -> CellData:
    """Flatten rollouts into per-cell rows.

    One masking rule: a row survives if its label, its confound and its features
    are all finite. Everything the target considers unscoreable — walls,
    unreachable cells, the objective's own cell, a whole episode that timed
    out — it expresses as ``NaN``, so this layer needs no maze knowledge.
    """
    xs, ys, confounds, episodes = [], [], [], []
    kept = total = 0
    for index, rollout in enumerate(rollouts):
        labels = target.labels(rollout)
        confound = target.confound(rollout)
        grid = feature(rollout)

        usable = np.isfinite(labels) & np.isfinite(confound).all(axis=-1) & np.isfinite(grid).all(axis=-1)
        total += labels.size
        kept += int(usable.sum())
        if not usable.any():
            continue

        xs.append(grid[usable])
        ys.append(labels[usable])
        confounds.append(confound[usable])
        episodes.append(np.full(int(usable.sum()), index))

    if not xs:
        raise ValueError(f"target {target.name!r} left no scoreable cells in {len(rollouts)} episodes")

    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    confound = np.concatenate(confounds).astype(np.float64)

    if np.any(confound[:, 0] > y + 1e-9):
        raise ValueError(
            f"target {target.name!r} declared a confound that exceeds its own labels; column 0 must be a "
            "lower bound, or the hard-cell subset is meaningless"
        )

    dropped = 0
    if drop_degenerate and x.shape[1] > 1:
        keep = x.std(axis=0) > DEGENERATE
        dropped = int((~keep).sum())
        if keep.any():
            x = x[:, keep]

    return CellData(
        x=x,
        y=y,
        confound=confound,
        episode=np.concatenate(episodes),
        mask_fraction=kept / max(total, 1),
        dropped_columns=dropped,
    )


def hard_cells(data: CellData, tau: float) -> np.ndarray:
    """Cells where the confound under-shoots the label by at least ``tau``.

    These are where a feature containing only the confound is *wrong* by
    construction, so they are where a real signal has to show itself.
    """
    return (data.y - data.confound[:, 0]) >= tau


def choose_l2(data: CellData, grid=(1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5), folds: int = 4, seed: int = 0):
    """Pick the penalty by held-out fit, grouping folds by episode.

    Returns ``(l2, interior)``. A winner at either end of the grid was clipped
    rather than chosen, and the caller should say so — the pilot's observation
    arm selected the grid maximum and nobody noticed.

    Selected on held-out R² of *recalibrated* predictions. Selecting on raw R²
    rewards under-dispersion, which manufactures the very shrinkage the
    calibration split exists to detect.
    """
    unique = np.unique(data.episode)
    assignment = np.random.default_rng(seed).permutation(len(unique)) % folds
    fold = np.array([dict(zip(unique, assignment))[e] for e in data.episode])

    scores = []
    for l2 in grid:
        per_fold = []
        for held in range(folds):
            train, test = fold != held, fold == held
            if not train.any() or not test.any():
                continue
            w, mean, std = fit_ridge(data.x[train], data.y[train], l2=l2)
            fitted = apply_linear(data.x[train], w, mean, std)
            scale, offset = metrics.affine_fit(data.y[train], fitted)
            predicted = scale * apply_linear(data.x[test], w, mean, std) + offset
            per_fold.append(metrics.r2(data.y[test], predicted))
        scores.append(float(np.mean(per_fold)) if per_fold else -np.inf)

    best = int(np.argmax(scores))
    return grid[best], 0 < best < len(grid) - 1


def fit_predict(train: CellData, test: CellData, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Predictions on ``test``, raw and recalibrated, plus the chosen penalty.

    Exposed separately because a paired comparison between two targets needs the
    predictions themselves, not a summary of them.
    """
    l2, interior = choose_l2(train, seed=seed)
    w, mean, std = fit_ridge(train.x, train.y, l2=l2)
    scale, offset = metrics.affine_fit(train.y, apply_linear(train.x, w, mean, std))
    raw = apply_linear(test.x, w, mean, std)
    return raw, scale * raw + offset, l2, interior


def field_probe(
    name: str,
    train: CellData,
    test: CellData,
    tau: float = 4.0,
    seed: int = 0,
    also: tuple[float, ...] = (2.0, 8.0),
) -> FieldResult:
    """Fit on ``train``, score on ``test``.

    Fitted on every usable cell and evaluated on the hard subset. Refitting on
    the subset would let the probe learn a subset-specific offset, which is an
    easier and different question.

    Predictions are recalibrated by a scale and offset fitted on *train*, which
    removes the shrinkage a penalty introduces without touching the ordering.
    The uncalibrated decomposition is reported alongside so the shrinkage stays
    visible rather than laundered.
    """
    raw, prediction, l2, interior = fit_predict(train, test, seed=seed)
    hard = hard_cells(test, tau)
    split = metrics.calibration(test.y[hard], raw[hard])
    interval = metrics.bootstrap_episodes(
        lambda rows: metrics.r2(test.y[rows][hard[rows]], prediction[rows][hard[rows]]), test.episode, seed=seed
    )

    stratum = test.confound[:, 0].astype(np.int64)
    return FieldResult(
        name=name if interior else f"{name} (l2 clipped)",
        hard_r2=metrics.r2(test.y[hard], prediction[hard]),
        hard_interval=interval,
        hard_shape_r2=split.shape_r2,
        pooled_r2=metrics.r2(test.y, prediction),
        partial_r=metrics.stratified_correlation(prediction, test.y, stratum),
        partial_r_within_episode=metrics.stratified_correlation(
            prediction, test.y, test.episode * 1000 + stratum
        ),
        slope=split.slope,
        bias=split.bias,
        mae=float(np.mean(np.abs(test.y - prediction))),
        n_samples=len(test.y),
        n_hard=int(hard.sum()),
        depth=test.x.shape[1],
        dropped_columns=test.dropped_columns,
        l2=l2,
        mask_fraction=test.mask_fraction,
        feature_norm=float(np.sqrt(np.mean(test.x**2))),
        sensitivity=tuple(
            (other, metrics.r2(test.y[rows], prediction[rows]))
            for other in also
            for rows in (hard_cells(test, other),)
        ),
    )
