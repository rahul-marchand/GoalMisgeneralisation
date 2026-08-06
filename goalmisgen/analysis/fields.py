"""Probing for a per-cell distance field.

The route probe asks a yes/no question — will the agent step here — and is
scored with AUC. This asks a magnitude question: how many steps is this cell
from the objective. That needs a regression fit and, more importantly, a
different set of defences.

The defence that matters is against **straight-line distance**. On real 11x11
levels, ``corr(bfs, manhattan) = 0.57`` and a linear model on straight-line
distance alone explains R² = 0.33 of the true field. A probe reading nothing but
free-space geometry — which an untrained convolutional tower supplies for free,
since iterating a 3x3 kernel spreads information isotropically — therefore
scores well enough to look like a finding. Pooled R² is the regression analogue
of the pooled negatives that invalidated the first distance-band result.

So the claim rests on two numbers, never on pooled R²:

**Detour R²** — accuracy restricted to cells where walls force a long way round
(``bfs - manhattan >= tau``). Straight-line distance is *wrong* there by
construction. Scored against the detour subset's own mean, because that subset
has a systematically larger target and the global mean would flatter every arm.

**Stratified partial correlation** — the exact generalisation of matched
negatives. Centre prediction and target within each integer straight-line
distance, then correlate the residuals: whatever a geometry-only feature could
have told you is removed by construction.

Deliberately plain functions. The probe-target protocol this will eventually
sit behind is a separate design question, and worth answering with a
measurement in hand rather than before one.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from goalmisgen.analysis import geometry
from goalmisgen.analysis.probes import apply_linear, fit_ridge

FeatureGrid = Callable[[object], np.ndarray]
"""Maps a rollout to a ``(height, width, depth)`` grid the probe reads per cell."""


@dataclasses.dataclass(frozen=True)
class FieldData:
    """Per-cell rows, masked identically across every arm so they are comparable."""

    x: np.ndarray
    """(n, depth) features."""

    y: np.ndarray
    """(n,) true shortest-path distance, in cells."""

    straight: np.ndarray
    """(n, 2) manhattan and chebyshev distance from the same source."""

    episode: np.ndarray
    """(n,) which rollout each row came from, for grouped resampling."""

    mask_fraction: float
    """Share of free cells that survived. Unreachable cells are dropped, and on
    these levels that is about one in six."""


@dataclasses.dataclass(frozen=True)
class FieldResult:
    name: str
    pooled_r2: float
    detour_r2: float
    detour_interval: tuple[float, float]
    partial_r: float
    mae: float
    n_samples: int
    n_detour: int
    depth: int
    l2: float
    mask_fraction: float
    activation_norm: float
    sensitivity: tuple[tuple[float, float], ...]
    """``(tau, detour R²)`` at thresholds either side of the reported one, so it
    is visible that the threshold was not chosen to suit the answer."""


def distance_target(rollout, feature_id: int, n_features: int = 2):
    """True field and its geometry-only null, for one objective.

    Returns ``(labels, manhattan, chebyshev)``, each ``(height, width)`` with
    ``NaN`` outside the reachable set.
    """
    observation = rollout.observation
    geometry.check_layout(observation, n_features)
    source = geometry.objective_cell(observation, feature_id)
    walls = geometry.blocking_walls(observation, feature_id, n_features)
    return (
        geometry.bfs_field(walls, source),
        geometry.manhattan_field(observation.shape[:2], source),
        geometry.chebyshev_field(observation.shape[:2], source),
    )


def field_dataset(rollouts, grid: FeatureGrid, feature_id: int, n_features: int = 2) -> FieldData:
    """Flatten rollouts into per-cell rows.

    Cells are kept when they are free and reachable. The objective's own cell is
    dropped: its distance is zero and it is a one-hot channel in the observation,
    so every arm would score it correctly and learn nothing. That is the same
    trap as scoring the route probe at step 0, where the observation baseline
    reached AUC 1.000 because the agent's cell is literally an input channel.
    """
    xs, ys, straights, episodes = [], [], [], []
    kept = total = 0
    for index, rollout in enumerate(rollouts):
        labels, manhattan, chebyshev = distance_target(rollout, feature_id, n_features)
        free = geometry.free_cells(rollout.observation)
        usable = free & np.isfinite(labels) & (labels > 0)

        total += int(free.sum())
        kept += int(usable.sum())

        xs.append(grid(rollout)[usable])
        ys.append(labels[usable])
        straights.append(np.stack([manhattan[usable], chebyshev[usable]], axis=1))
        episodes.append(np.full(int(usable.sum()), index))

    return FieldData(
        x=np.concatenate(xs).astype(np.float64),
        y=np.concatenate(ys).astype(np.float64),
        straight=np.concatenate(straights).astype(np.float64),
        episode=np.concatenate(episodes),
        mask_fraction=kept / max(total, 1),
    )


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    """Variance explained, against the mean of *these* rows.

    Evaluating a subset against the global mean would credit an arm for the
    subset simply having a larger target than average.
    """
    if len(y) < 2:
        return float("nan")
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def detour_cells(data: FieldData, tau: float) -> np.ndarray:
    """Cells where walls force a detour of at least ``tau`` over the straight line."""
    return (data.y - data.straight[:, 0]) >= tau


def stratified_correlation(prediction: np.ndarray, y: np.ndarray, stratum: np.ndarray) -> float:
    """Correlation after centring both sides within each stratum.

    Matching negatives by distance is the classification form of this: compare
    only within groups the confound cannot separate.

    Strata with a single member are dropped rather than centred to zero. A lone
    point carries no within-stratum information, and keeping it would pad the
    sample with rows that can only dilute the estimate.

    Zero residual variance returns 0.0, not ``NaN``: it means the confound
    accounted for everything, which is a result. ``NaN`` is reserved for having
    no comparable rows at all.
    """
    p = prediction.astype(np.float64).copy()
    t = y.astype(np.float64).copy()

    usable = np.zeros(len(p), dtype=bool)
    for value in np.unique(stratum):
        rows = stratum == value
        if rows.sum() < 2:
            continue
        p[rows] -= p[rows].mean()
        t[rows] -= t[rows].mean()
        usable |= rows

    if usable.sum() < 2:
        return float("nan")
    if p[usable].std() < 1e-12 or t[usable].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p[usable], t[usable])[0, 1])


def bootstrap_episodes(statistic, episode: np.ndarray, resamples: int = 200, seed: int = 0, level: float = 0.95):
    """Resample whole episodes, never cells.

    Cells in an episode share one maze and one field, so treating them as
    independent understates the uncertainty — the same reason ``auc_interval``
    resamples episodes.
    """
    unique = np.unique(episode)
    index_by_episode = {value: np.flatnonzero(episode == value) for value in unique}

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([index_by_episode[unique[i]] for i in chosen])
        value = statistic(rows)
        if np.isfinite(value):
            values.append(value)

    if not values:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def choose_l2(data: FieldData, grid=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3), folds: int = 4, seed: int = 0) -> float:
    """Pick the penalty by held-out fit, grouping folds by episode.

    Not a detail: a 96-dimensional arm and a 2-dimensional one under a single
    fixed penalty get very different effective shrinkage, and the *positive*
    control could lose to the trained arm purely from that mismatch — which
    would read as the rig being broken.

    Chosen on pooled held-out R², deliberately not on the detour number the
    result is reported against.
    """
    unique = np.unique(data.episode)
    assignment = np.random.default_rng(seed).permutation(len(unique)) % folds
    fold_of = dict(zip(unique, assignment))
    fold = np.array([fold_of[e] for e in data.episode])

    best, best_score = grid[0], -np.inf
    for l2 in grid:
        scores = []
        for held in range(folds):
            train, test = fold != held, fold == held
            if not train.any() or not test.any():
                continue
            w, mean, std = fit_ridge(data.x[train], data.y[train], l2=l2)
            scores.append(r2(data.y[test], apply_linear(data.x[test], w, mean, std)))
        score = float(np.mean(scores)) if scores else -np.inf
        if score > best_score:
            best, best_score = l2, score
    return best


def field_probe(
    name: str,
    train: FieldData,
    test: FieldData,
    tau: float = 4.0,
    seed: int = 0,
    also: tuple[float, ...] = (2.0, 8.0),
) -> FieldResult:
    """Fit on ``train``, score on ``test``.

    Fitted on *all* usable cells and only evaluated on detour cells. Refitting on
    the detour subset would let the probe learn a detour-specific offset, which
    is a different and easier question.
    """
    l2 = choose_l2(train, seed=seed)
    w, mean, std = fit_ridge(train.x, train.y, l2=l2)
    prediction = apply_linear(test.x, w, mean, std)

    detour = detour_cells(test, tau)
    interval = bootstrap_episodes(
        lambda rows: r2(test.y[rows][detour[rows]], prediction[rows][detour[rows]]),
        test.episode,
        seed=seed,
    )

    return FieldResult(
        name=name,
        pooled_r2=r2(test.y, prediction),
        detour_r2=r2(test.y[detour], prediction[detour]),
        detour_interval=interval,
        partial_r=stratified_correlation(prediction, test.y, test.straight[:, 0].astype(np.int64)),
        mae=float(np.mean(np.abs(test.y - prediction))),
        n_samples=len(test.y),
        n_detour=int(detour.sum()),
        depth=test.x.shape[1],
        l2=l2,
        mask_fraction=test.mask_fraction,
        activation_norm=float(np.sqrt(np.mean(test.x**2))),
        sensitivity=tuple(
            (other, r2(test.y[rows], prediction[rows]))
            for other in also
            for rows in (detour_cells(test, other),)
        ),
    )
