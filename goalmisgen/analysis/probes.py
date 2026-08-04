"""Linear probes over per-cell activations.

A probe is deliberately *linear* and applied identically at every cell — the
1x1-convolution setup from the planning-interpretability work. The restriction
is the point: a linear readout cannot solve a maze, so if it can predict which
cells the agent will step on, the route must already be present in the
representation rather than being computed by the probe.

Which makes the **observation baseline** essential. The same probe fitted to the
raw observation has access to the walls and the goals but no plan; if it scores
as well as the activation probe, nothing has been demonstrated. The gap between
them is the result.

Fitted with plain gradient descent on the logistic loss — scikit-learn is not a
dependency, and a per-cell binary readout does not need one.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import numpy as np

from goalmisgen.envs.observation import WALL_CHANNEL

ProbeSource = Literal["features", "observation"]
"""Which per-cell grid a probe reads. Mistyping a plain string silently
probed the observation and reported a plausible AUC."""


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """How well a linear readout recovers the label, and how to beat chance."""

    auc: float
    """Ranking accuracy, insensitive to the positive-class rate."""

    accuracy: float
    balanced_accuracy: float
    positive_rate: float
    """Fraction of cells labelled positive - the floor a trivial probe achieves."""

    n_samples: int
    n_features: int

    def __str__(self) -> str:
        return (
            f"AUC {self.auc:.3f}   balanced acc {self.balanced_accuracy:.3f}   "
            f"acc {self.accuracy:.3f}  (positives {self.positive_rate:.1%}, "
            f"n={self.n_samples:,} cells, d={self.n_features})"
        )


def _cells(rollouts: Sequence, source: ProbeSource = "features", mask_walls: bool = True):
    """Flatten rollouts into per-cell rows: features, label, arrival step, distance."""
    xs, ys, steps, distances = [], [], [], []
    for r in rollouts:
        grid = r.features if source == "features" else r.observation
        free = r.observation[:, :, WALL_CHANNEL] < 0.5
        keep = free if mask_walls else np.ones_like(free, dtype=bool)
        xs.append(grid[keep])
        ys.append(r.visited[keep])
        steps.append(r.visit_step[keep])
        distances.append(r.distance[keep])
    return (
        np.concatenate(xs).astype(np.float64),
        np.concatenate(ys).astype(np.float64),
        np.concatenate(steps).astype(np.int64),
        np.concatenate(distances).astype(np.int64),
    )


def cell_dataset(rollouts, source: ProbeSource = "features", mask_walls: bool = True):
    """Flatten rollouts into (cell, feature) rows with an on-route label.

    Wall cells are dropped by default: the agent can never stand on one, so
    including them inflates every score with trivially-negative examples.
    """
    x, y, _, _ = _cells(rollouts, source, mask_walls)
    return x, y


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.5, l2: float = 1e-3):
    """Logistic regression by gradient descent on standardised inputs."""
    mean, std = x.mean(0), x.std(0) + 1e-8
    z = (x - mean) / std
    z = np.hstack([z, np.ones((len(z), 1))])

    w = np.zeros(z.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        grad = z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * grad
    return w, mean, std


def apply_logistic(x: np.ndarray, w, mean, std) -> np.ndarray:
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    return 1.0 / (1.0 + np.exp(-z @ w))


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


@dataclasses.dataclass(frozen=True)
class DistanceBand:
    """How well the probe finds cells the agent reaches this many steps later."""

    step: int
    auc: float
    n_positive: int


def probe_by_distance(train, test, source: ProbeSource = "features", max_step: int = 12) -> list[DistanceBand]:
    """Score one probe separately at each distance along the route.

    A probe can reach a high overall AUC by finding only the cell the agent is
    about to move onto, which would make it a move predictor wearing a plan's
    clothes. Splitting by arrival step separates the two: a plan stays decodable
    at the far end of the route, an imminent action does not.

    Negatives are **distance-matched**: a cell reached at step ``k`` is scored
    only against cells the agent never visits that are also ``k`` steps away.
    Pooling all never-visited cells instead makes the comparison "distance k
    versus every distance", which a feature encoding nothing but distance-from-
    agent answers correctly — that control scores 0.99 down to 0.56 across the
    bands with no plan in it, and exactly 0.50 once the negatives are matched.
    """
    x_train, y_train, _, _ = _cells(train, source)
    x_test, y_test, steps, distances = _cells(test, source)

    w, mean, std = fit_logistic(x_train, y_train)
    scores = apply_logistic(x_test, w, mean, std)
    unvisited = y_test == 0

    bands = []
    for step in range(max_step + 1):
        selected = steps == step
        matched = unvisited & (distances == step)
        # Both sides need enough cells for the comparison to mean anything.
        if selected.sum() < 20 or matched.sum() < 20:
            continue
        combined = np.concatenate([scores[selected], scores[matched]])
        labels = np.concatenate([np.ones(int(selected.sum())), np.zeros(int(matched.sum()))])
        bands.append(DistanceBand(step, roc_auc(labels, combined), int(selected.sum())))
    return bands


def probe(train, test, source: ProbeSource = "features") -> ProbeResult:
    """Fit on ``train`` rollouts, score on held-out ``test`` rollouts."""
    x_train, y_train = cell_dataset(train, source)
    x_test, y_test = cell_dataset(test, source)

    w, mean, std = fit_logistic(x_train, y_train)
    scores = apply_logistic(x_test, w, mean, std)
    predicted = scores >= 0.5

    positives = y_test == 1
    balanced = 0.5 * (predicted[positives].mean() + (~predicted[~positives]).mean())
    return ProbeResult(
        auc=roc_auc(y_test, scores),
        accuracy=float((predicted == y_test).mean()),
        balanced_accuracy=float(balanced),
        positive_rate=float(y_test.mean()),
        n_samples=len(y_test),
        n_features=x_test.shape[1],
    )
