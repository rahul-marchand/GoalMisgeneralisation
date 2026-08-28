"""Per-level scalar probes: reading the route model's own distance estimates.

UtilityRule.md part 3 ends in a gauge: behaviour alone cannot say whether the
per-level residual xi is a wobbling threshold or a misread of the distance
gap, because only the comparison's sign is observable. Breaking the gauge
needs a second, independent measurement of the model's gap estimate. These
probes provide it: ridge readouts of ``d_rich``, ``d_poor`` and their
difference from the prefix residual stream, trained against the solver's
ground truth on levels disjoint from the evaluation set.

Two things distinguish this from :mod:`goalmisgen.offline.probe`:

- The unit is the *level*, not the cell: one feature vector and one scalar
  target per maze, taken at a named site - SEP, the agent's cell, the two
  objective cells, or the mean over all cells.
- SEP is kept. ``cell_residuals`` slices it off, but the first action's logit
  is produced at SEP, so it is the one position guaranteed to hold whatever
  the decision consumes.

The probe is trained on true distances, never on choice. The model's own
misread then lives in the probe's *residual*, and whether that residual
predicts choice within a fixed ``(d_rich, d_poor)`` cell - where the true
target is constant, so the residual is all the probe has left - is the
experiment. An imperfect probe weakens that signal but cannot fake it.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.probes import apply_linear, fit_ridge
from goalmisgen.offline.demos import NO_ACTION, DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM

SITES: tuple[str, ...] = ("sep", "agent", "objectives", "cells_mean")
"""Where in the prefix a level's feature vector is read.

``objectives`` concatenates the richer objective's cell before the poorer
one's, so the probe does not have to learn which colour is which.
"""


@functools.lru_cache(maxsize=8)
def _prefix_residuals_fn(config: ModelConfig):
    model = RoutePrefixLM(config)

    @jax.jit
    def residuals(params, observations):
        actions = jnp.full((observations.shape[0], config.max_actions), NO_ACTION, dtype=jnp.int32)
        _, streams = model.apply(params, observations, actions)
        return jnp.stack([s[:, : config.prefix_length] for s in streams], axis=0)

    return residuals


@dataclasses.dataclass(frozen=True)
class GapTargets:
    """Ground truth per level, from the solver via the demonstration set."""

    d_rich: np.ndarray  # (B,) int - distance to the richer objective
    d_poor: np.ndarray  # (B,) int
    richer: np.ndarray  # (B,) int - objective index of the richer
    valid: np.ndarray  # (B,) bool - both objectives reachable

    @property
    def gap(self) -> np.ndarray:
        return self.d_rich - self.d_poor


def gap_targets(demos: DemoSet, indices: np.ndarray) -> GapTargets:
    indices = np.asarray(indices)
    values = np.asarray(demos.values)[indices]
    distances = np.asarray(demos.distances)[indices].astype(np.int64)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(indices))
    d_rich = distances[rows, richer]
    d_poor = distances[rows, 1 - richer]
    return GapTargets(d_rich=d_rich, d_poor=d_poor, richer=richer, valid=(d_rich >= 0) & (d_poor >= 0))


def collect_site_features(
    model: RoutePrefixLM,
    params,
    demos: DemoSet,
    indices: np.ndarray,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """``{site: (B, n_depths, width)}`` float32, extracted chunk by chunk.

    The full stream for 50k levels is ~16 GB, so it is never materialised;
    each chunk's ``(n_depths, b, prefix_length, d_model)`` block is reduced to
    the site vectors and dropped.
    """
    cfg = model.config
    indices = np.asarray(indices)
    fn = _prefix_residuals_fn(cfg)
    targets = gap_targets(demos, indices)
    agent = np.asarray(demos.agent)[indices].astype(np.int64)
    positions = np.asarray(demos.positions)[indices].astype(np.int64)
    rich_pos = positions[np.arange(len(indices)), targets.richer]
    poor_pos = positions[np.arange(len(indices)), 1 - targets.richer]

    def token(rc: np.ndarray) -> np.ndarray:
        return rc[:, 0] * cfg.size + rc[:, 1]

    out: dict[str, list[np.ndarray]] = {site: [] for site in SITES}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        rows = slice(start, start + len(chunk))
        streams = np.asarray(fn(params, jnp.asarray(demos.observations(chunk))))  # (depths, b, prefix, width)
        pick = np.arange(len(chunk))
        out["sep"].append(streams[:, :, -1].transpose(1, 0, 2))
        out["agent"].append(streams[:, pick, token(agent[rows])].transpose(1, 0, 2))
        out["objectives"].append(
            np.concatenate(
                [
                    streams[:, pick, token(rich_pos[rows])],
                    streams[:, pick, token(poor_pos[rows])],
                ],
                axis=-1,
            ).transpose(1, 0, 2)
        )
        out["cells_mean"].append(streams[:, :, : cfg.n_cells].mean(axis=2).transpose(1, 0, 2))
    return {site: np.concatenate(chunks, axis=0) for site, chunks in out.items()}


def flatten_depths(features: np.ndarray, layer: int | None) -> np.ndarray:
    """``(B, width)`` for one depth, or every depth concatenated for ``None``."""
    if layer is None:
        return features.reshape(len(features), -1)
    return features[:, layer]


@dataclasses.dataclass(frozen=True)
class GapProbeResult:
    r2: float
    mae: float
    residual_std: float
    l2: float
    predictions: np.ndarray  # (B_eval,) - so residuals can be taken against choice

    @property
    def summary(self) -> str:
        return f"r2 {self.r2:+.3f}  mae {self.mae:.2f}  resid sd {self.residual_std:.2f}  l2 {self.l2:g}"


def fit_gap_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    l2_grid: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0),
    val_fraction: float = 0.2,
    seed: int = 0,
) -> GapProbeResult:
    """Ridge with ``l2`` chosen on a held-back slice of the training levels.

    The target stays in steps, as :func:`fit_ridge` leaves it, so the residual
    standard deviation reads directly against the behavioural xi spread.
    """
    order = np.random.default_rng(seed).permutation(len(x_train))
    cut = max(1, int(len(order) * val_fraction))
    val, fit = order[:cut], order[cut:]

    def val_mse(l2: float) -> float:
        w, mean, std = fit_ridge(x_train[fit], y_train[fit], l2=l2)
        return float(np.mean((apply_linear(x_train[val], w, mean, std) - y_train[val]) ** 2))

    best = min(l2_grid, key=val_mse)
    w, mean, std = fit_ridge(x_train, y_train, l2=best)
    predictions = apply_linear(x_eval, w, mean, std)
    residual = predictions - y_eval
    total = float(np.var(y_eval))
    return GapProbeResult(
        r2=1.0 - float(np.mean(residual**2)) / total if total > 0 else float("nan"),
        mae=float(np.mean(np.abs(residual))),
        residual_std=float(np.std(residual)),
        l2=float(best),
        predictions=predictions,
    )


@dataclasses.dataclass(frozen=True)
class WithinCellResult:
    """Does the probe's residual predict choice once the true distances are fixed?

    Within one exact ``(d_rich, d_poor)`` cell the probe's target is a
    constant, so its within-cell variation is pure residual: the model's own
    misread plus probe noise. ``auc`` is the pair-weighted mean of per-cell
    AUCs of ``-prediction`` against taking the richer objective; ``r2`` is the
    within-cell variance of choice a linear readout of the residual explains,
    the same statistic the 21 hand-built level features scored ~0.01 on.
    """

    auc: float
    r2: float
    n_cells: int
    n_levels: int

    @property
    def summary(self) -> str:
        return f"auc {self.auc:.3f}  r2 {self.r2:.4f}  cells {self.n_cells}  levels {self.n_levels:,}"


def cell_members(d_rich: np.ndarray, d_poor: np.ndarray, min_n: int = 40) -> list[np.ndarray]:
    """Index groups sharing an exact ``(d_rich, d_poor)`` pair, small cells dropped."""
    cells = []
    for key in sorted(set(zip(d_rich.tolist(), d_poor.tolist()))):
        members = np.flatnonzero((d_rich == key[0]) & (d_poor == key[1]))
        if len(members) >= min_n:
            cells.append(members)
    return cells


def centre_within_cells(values: np.ndarray, cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``values`` minus its cell mean, and the concatenated member indices.

    Within a cell the probe's target is constant, so centred predictions *are*
    the probe's residuals - which is what makes two probes' agreement here a
    reliability rather than a restatement of the shared target.
    """
    centred = [values[members] - values[members].mean() for members in cells]
    members = np.concatenate(cells) if cells else np.empty(0, dtype=np.int64)
    return (np.concatenate(centred) if centred else np.empty(0)), members


def within_cell_choice(
    d_rich: np.ndarray,
    d_poor: np.ndarray,
    score: np.ndarray,
    choice: np.ndarray,
    min_n: int = 40,
) -> WithinCellResult:
    """``score`` is the probed gap: higher means the rich objective reads further."""
    from goalmisgen.analysis.probes import roc_auc

    cells = cell_members(d_rich, d_poor, min_n=min_n)

    aucs, weights = [], []
    centred_score, centred_choice = [], []
    n_levels = 0
    for members in cells:
        y = choice[members].astype(np.float64)
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        n_levels += len(members)
        centred_score.append(score[members] - score[members].mean())
        centred_choice.append(y - y.mean())
        if n_pos and n_neg:
            aucs.append(roc_auc(y, -score[members]))
            weights.append(n_pos * n_neg)

    centred_score = np.concatenate(centred_score) if centred_score else np.empty(0)
    centred_choice = np.concatenate(centred_choice) if centred_choice else np.empty(0)
    if len(centred_score) and centred_score.std() > 0 and centred_choice.std() > 0:
        r = float(np.corrcoef(centred_score, centred_choice)[0, 1])
    else:
        r = float("nan")
    return WithinCellResult(
        auc=float(np.average(aucs, weights=weights)) if aucs else float("nan"),
        r2=r * r if np.isfinite(r) else float("nan"),
        n_cells=len(cells),
        n_levels=n_levels,
    )
