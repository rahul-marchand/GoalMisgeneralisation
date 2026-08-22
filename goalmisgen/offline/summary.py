"""The SEP token: the one position that has to summarise the maze.

Every per-cell probe in this project asks whether a quantity is spread over a
grid. The route model offers a site the DRC does not have, and the difference
matters for a *scalar* like "how far is objective 0".

The DRC's actor is a Dense layer over the flattened recurrent grid, so a scalar
held once and a field averaged over cells reach the policy by the same path and
no probe can separate them. Here they are different token positions. SEP is the
position the first action is predicted from, and it is the only prefix position
that is not about one cell; a distance that reads there but not per-cell, or
per-cell but not there, is a fact about where the model keeps the quantity
rather than about how a readout was set up.

The probe is therefore an ordinary ridge over episodes rather than over cells,
and it takes the same three controls the field probes take
(:func:`goalmisgen.analysis.targets.controls`): a target handed over as its own
feature, a no-computation geometric null, and the target attached to a different
episode.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis import metrics
from goalmisgen.analysis.probes import apply_linear, fit_ridge
from goalmisgen.offline.demos import NO_ACTION
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


@functools.lru_cache(maxsize=8)
def _sep_fn(config: ModelConfig):
    model = RoutePrefixLM(config)

    @jax.jit
    def sep(params, observations):
        actions = jnp.full((observations.shape[0], config.max_actions), NO_ACTION, dtype=jnp.int32)
        _, streams = model.apply(params, observations, actions)
        return jnp.stack([s[:, config.n_cells] for s in streams], axis=0)

    return sep


def sep_residuals(model: RoutePrefixLM, params, observations: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """``(n_layers + 1, B, d_model)``: the SEP position at every depth.

    Captured with no action tokens present, like the per-cell grids. Attention
    over the prefix is bidirectional, so SEP has seen every maze token, and it
    cannot have seen an action: this is the model's summary of the level and
    nothing else.
    """
    fn = _sep_fn(model.config)
    chunks = [
        np.asarray(fn(params, jnp.asarray(observations[start : start + batch_size])))
        for start in range(0, len(observations), batch_size)
    ]
    return np.concatenate(chunks, axis=1)


@functools.lru_cache(maxsize=16)
def _sep_edited_fn(config: ModelConfig, edit_depth: int):
    model = RoutePrefixLM(config)

    @jax.jit
    def sep(params, observations, edit):
        actions = jnp.full((observations.shape[0], config.max_actions), NO_ACTION, dtype=jnp.int32)
        _, streams = model.apply(params, observations, actions, edit, edit_depth)
        return jnp.stack([s[:, config.n_cells] for s in streams], axis=0)

    return sep


def sep_residuals_edited(
    model: RoutePrefixLM, params, observations: np.ndarray, edit: np.ndarray, edit_depth: int, batch_size: int = 512
) -> np.ndarray:
    """:func:`sep_residuals` of the *written* network.

    Which is how "does the edit survive to where the head reads it" is asked.
    Blocks after ``edit_depth`` can rebuild SEP from the maze tokens, and those
    are untouched by a write at this site, so a decision that does not move may
    be a decision that never saw the write.
    """
    fn = _sep_edited_fn(model.config, edit_depth)
    chunks = [
        np.asarray(
            fn(
                params,
                jnp.asarray(observations[start : start + batch_size]),
                jnp.asarray(edit[start : start + batch_size]),
            )
        )
        for start in range(0, len(observations), batch_size)
    ]
    return np.concatenate(chunks, axis=1)


@dataclasses.dataclass(frozen=True)
class ScalarResult:
    """One scalar probe, in the units of the thing it decodes."""

    name: str
    r2: float
    interval: tuple[float, float]
    partial_r: float
    """Correlation with the truth *within* a stratum of the free-space confound.

    The headline, for the reason ``005`` makes it the headline of the field
    table. Manhattan distance alone explains a third of the variance in a true
    field without solving anything, so a raw R2 rewards a probe for rediscovering
    geometry. Stratified, only maze-aware structure survives - and a residual
    whose raw R2 sits *below* the null's is a site that carries less than the
    free-space number, however respectable its own R2 looks."""

    mae: float
    """Mean absolute error in cells, which is the number to read: an R² can be
    respectable while the errors are larger than the distances being compared."""

    slope: float
    n: int
    depth: int

    def __str__(self) -> str:
        return (
            f"{self.name:>34}{self.r2:>9.3f}{f'[{self.interval[0]:.3f},{self.interval[1]:.3f}]':>18}"
            f"{self.partial_r:>9.3f}{self.mae:>7.2f}{self.slope:>8.2f}{self.depth:>5}{self.n:>7,}"
        )


def choose_l2(
    x: np.ndarray,
    y: np.ndarray,
    grid=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5),
    folds: int = 4,
    seed: int = 0,
) -> float:
    """The penalty with the best held-out R2, by k-fold on the training episodes.

    A scalar probe reads ``d_model`` features from a few hundred episodes, where
    a field probe reads them from tens of thousands of cells. At a fixed penalty
    that difference alone decides the answer: the same site scores below zero on
    64 episodes and respectably on a thousand, and neither number is about the
    model. Chosen here for the same reason ``fields.choose_l2`` exists.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)

    best, best_score = grid[0], -np.inf
    for l2 in grid:
        scores = []
        for index in range(folds):
            held = parts[index]
            kept = np.concatenate([parts[other] for other in range(folds) if other != index])
            weights, mean, std = fit_ridge(x[kept], y[kept], l2=l2)
            scores.append(metrics.r2(y[held], apply_linear(x[held], weights, mean, std)))
        score = float(np.mean(scores))
        if score > best_score:
            best, best_score = l2, score
    return best


def fit_scalar(train_x: np.ndarray, train_y: np.ndarray, l2: float | None = None, seed: int = 0):
    """Ridge on the episodes whose target exists. Returns ``(w, mean, std)``.

    Separate from :func:`scalar_probe` because the paired own/other comparison
    needs predictions for *every* test episode, aligned by index, and a probe
    that drops rows internally cannot hand those back. ``l2=None`` chooses the
    penalty by cross-validation on the training episodes.
    """
    ok = np.isfinite(train_y)
    penalty = choose_l2(train_x[ok], train_y[ok], seed=seed) if l2 is None else l2
    return fit_ridge(train_x[ok], train_y[ok], l2=penalty)


def scalar_probe(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    confound: np.ndarray | None = None,
    l2: float | None = None,
    seed: int = 0,
) -> ScalarResult:
    """Ridge from one vector per episode to one number per episode.

    Episodes with no target - an objective that cannot be reached around the
    other - are dropped from both sides rather than filled, since a fill is a
    number the probe can learn.
    """
    test_ok = np.isfinite(test_y)
    weights, mean, std = fit_scalar(train_x, train_y, l2=l2, seed=seed)
    prediction = apply_linear(test_x[test_ok], weights, mean, std)
    truth = test_y[test_ok]

    episodes = np.arange(len(truth))
    interval = metrics.bootstrap_episodes(
        lambda rows: metrics.r2(truth[rows], prediction[rows]), episodes, seed=seed
    )
    slope, _ = metrics.affine_fit(truth, prediction)
    stratum = np.zeros(len(truth), dtype=np.int64) if confound is None else confound[test_ok].astype(np.int64)
    return ScalarResult(
        name=name,
        r2=metrics.r2(truth, prediction),
        interval=interval,
        partial_r=metrics.stratified_correlation(prediction, truth, stratum),
        mae=float(np.mean(np.abs(truth - prediction))),
        slope=slope,
        n=int(test_ok.sum()),
        depth=test_x.shape[1],
    )


def own_and_other(
    predictions: dict[int, np.ndarray],
    truths: dict[int, np.ndarray],
    reached: np.ndarray,
    seed: int = 0,
) -> tuple[float, float, tuple[float, float]]:
    """Absolute error for the objective an episode reached against the other one.

    The keying question of ``005``, asked of a scalar. One probe per objective,
    each decoding a quantity that never changes identity, and the outcome enters
    by splitting episodes - so every episode contributes to both sides and the
    between-level variance cancels in the paired interval.

    Returns ``(own, other, interval on other - own)``. A positive difference
    means the objective the model went to is held more precisely than the one it
    passed up, which is the "commits first, measures second" answer.
    """
    usable = np.array(
        [
            index
            for index in range(len(reached))
            if reached[index] in truths
            and all(np.isfinite(truths[feature][index]) for feature in truths)
        ]
    )
    if not len(usable):
        raise ValueError("no episode has a finite distance to both objectives")

    own_error = np.array([abs(predictions[reached[i]][i] - truths[reached[i]][i]) for i in usable])
    other_error = np.array(
        [
            abs(predictions[1 - reached[i]][i] - truths[1 - reached[i]][i])
            for i in usable
        ]
    )
    episodes = np.arange(len(usable))
    low, high = metrics.bootstrap_paired(
        lambda rows: float(other_error[rows].mean()),
        lambda rows: float(own_error[rows].mean()),
        episodes,
        seed=seed,
    )
    return float(own_error.mean()), float(other_error.mean()), (low, high)
