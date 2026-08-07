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
import itertools
from typing import Callable, Literal, Sequence

import numpy as np

from goalmisgen.envs.observation import WALL_CHANNEL

ProbeSource = Literal["features", "observation"]
"""Which per-cell grid a probe reads. Mistyping a plain string silently
probed the observation and reported a plausible AUC."""


@dataclasses.dataclass(frozen=True)
class Feature:
    """A named per-cell grid the probe reads: ``(height, width, depth)``.

    Named because every arm of an experiment is identified by one, and an
    unnamed callable in a results table is unreadable. Free to evaluate, unlike
    a :class:`~goalmisgen.analysis.activations.Capture`, which costs a rollout —
    the two are separate for that reason.
    """

    name: str
    grid: Callable[[object], np.ndarray]

    def __call__(self, rollout) -> np.ndarray:
        return self.grid(rollout)


def layer_slice(feature: Feature, index: int, n_layers: int) -> Feature:
    """One layer's share of a per-cell state that concatenates several.

    Which layer a probe reads is not a detail when the question is causal. A
    DRC's actor sees only the *last* recurrent layer's hidden state, so a
    direction spread evenly over all of them aims most of itself at components
    the policy never reads.
    """

    def sliced(rollout) -> np.ndarray:
        grid = feature(rollout)
        depth = grid.shape[-1]
        if depth % n_layers:
            raise ValueError(f"{depth} channels do not divide into {n_layers} layers")
        width = depth // n_layers
        return grid[..., index * width : (index + 1) * width]

    return Feature(f"{feature.name}[layer {index}]", sliced)


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


def _design(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardised inputs with a bias column appended."""
    return np.hstack([(x - mean) / std, np.ones((len(x), 1))])


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.5, l2: float = 1e-3):
    """Logistic regression by gradient descent on standardised inputs."""
    mean, std = x.mean(0), x.std(0) + 1e-8
    z = _design(x, mean, std)

    w = np.zeros(z.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        grad = z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * grad
    return w, mean, std


def apply_logistic(x: np.ndarray, w, mean, std) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_design(x, mean, std) @ w))


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 1.0):
    """Ridge regression in closed form, for probes with a continuous target.

    ``fit_logistic`` uses gradient descent only because the logistic loss has no
    closed form; least squares does, and on a matrix this small solving it
    exactly is both faster and one fewer thing to tune.

    The target is left in its own units — distances stay in cells, so a mean
    absolute error is directly readable — and the bias is not penalised, so
    shrinkage cannot bias predictions toward zero.
    """
    mean, std = x.mean(0), x.std(0) + 1e-8
    z = _design(x, mean, std)

    penalty = l2 * np.eye(z.shape[1])
    penalty[-1, -1] = 0.0  # never shrink the intercept
    w = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return w, mean, std


def apply_linear(x: np.ndarray, w, mean, std) -> np.ndarray:
    return _design(x, mean, std) @ w


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


def auc_interval(
    train,
    test,
    source: ProbeSource = "features",
    resamples: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap the AUC by resampling whole episodes.

    Cells are not independent: every cell in an episode shares one maze and one
    route. Treating a few thousand cells as a few thousand samples understates
    the uncertainty by about 1.6x, and reads as a far larger experiment than the
    hundred-odd episodes it actually is.
    """
    x_train, y_train = cell_dataset(train, source)
    w, mean, std = fit_logistic(x_train, y_train)

    episodes = []
    for rollout in test:
        x, y = cell_dataset([rollout], source)
        episodes.append((apply_logistic(x, w, mean, std), y))

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(episodes), len(episodes))
        auc = roc_auc(
            np.concatenate([episodes[i][1] for i in chosen]),
            np.concatenate([episodes[i][0] for i in chosen]),
        )
        if not np.isnan(auc):
            values.append(auc)

    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


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


def fit_multinomial(x: np.ndarray, y: np.ndarray, n_classes: int, steps: int = 600, lr: float = 0.5, l2: float = 1e-3):
    """Softmax regression by gradient descent, for a per-cell class label.

    The multi-class twin of :func:`fit_logistic`, and it exists for the same
    reason that one does: a per-cell readout does not justify a dependency.

    Returns weights of shape ``(depth + 1, n_classes)`` — the last row is the
    bias, unpenalised, so a rare class is not shrunk toward never being
    predicted.
    """
    mean, std = x.mean(0), x.std(0) + 1e-8
    z = _design(x, mean, std)
    one_hot = np.eye(n_classes)[y.astype(np.int64)]

    w = np.zeros((z.shape[1], n_classes))
    for _ in range(steps):
        logits = z @ w
        logits -= logits.max(axis=1, keepdims=True)  # softmax is shift-invariant; this keeps exp finite
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        penalty = l2 * np.vstack([w[:-1], np.zeros((1, n_classes))])
        w -= lr * (z.T @ (probabilities - one_hot) / len(y) + penalty)
    return w, mean, std


def apply_multinomial(x: np.ndarray, w, mean, std) -> np.ndarray:
    """Class probabilities, ``(n_samples, n_classes)``."""
    logits = _design(x, mean, std) @ w
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _max_margin_direction(differences: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit vector maximising ``min_j differences[j] . delta``, and that minimum.

    The optimum is the minimum-norm point of the convex hull, so candidates come
    from enumerating the hull's faces — on each, the equality-constrained
    problem ``min lam^T G lam, sum lam = 1`` has a closed form. Exponentiated
    gradient was tried first and was wrong in the worst available way: it
    returned directions whose reported margin was positive while the true margin
    against one rival was negative, so the edit did nothing and the number
    beside it said otherwise.

    Every candidate is then scored on the **actual** objective and the best is
    returned. That is not belt-and-braces: a face whose Gram matrix is singular
    is exactly the case where a class is unwritable, and it is also the case the
    closed form cannot solve, so trusting the algebra would skip the evidence
    and report a comfortable margin from some other face.
    """
    count = len(differences)
    if count > 12:
        raise ValueError(f"face enumeration is exponential; {count} rival classes is too many")

    gram = differences @ differences.T
    best_direction, best_margin = None, -np.inf
    for size in range(1, count + 1):
        for face in itertools.combinations(range(count), size):
            ones = np.ones(size)
            weights, *_ = np.linalg.lstsq(gram[np.ix_(face, face)], ones, rcond=None)
            total = ones @ weights
            if total <= 0:
                continue
            point = (weights / total) @ differences[list(face)]
            norm = float(np.linalg.norm(point))
            if norm < 1e-12:
                continue
            direction = point / norm
            margin = float((differences @ direction).min())
            if margin > best_margin:
                best_direction, best_margin = direction, margin
    if best_direction is None:
        raise ValueError("no candidate direction: every face of the hull collapsed to the origin")
    return best_direction, best_margin


def class_directions(w: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors in raw activation space that make each class win, and by how much.

    Returns ``(directions, margins)``, both indexed by class.

    Two corrections to the obvious construction, and the second is not a detail.

    The probe is fitted on **standardised** inputs, so a row of ``w`` is not a
    direction in the space the activations live in: the logit for class ``k`` is
    ``w_k . (x - mean) / std``, whose gradient with respect to ``x`` is
    ``w_k / std``. Adding ``w_k`` itself steers a differently scaled space and
    still produces a plausible number.

    And adding ``w_k / std`` — the planning-interpretability paper's ``g + w_k``,
    corrected for scaling — **does not reliably make the probe read class k**. It
    raises class ``k``'s logit, but it can raise another class's faster: if
    ``v_j . v_k > |v_k|^2`` for some other class, then ``j`` wins at every
    magnitude, and no scale rescues it. On a four-class synthetic problem one
    class in four was unreachable this way at any alpha up to 1000.

    So the direction here maximises the *minimum* margin over all rival classes,
    ``max_delta min_j (v_k - v_j) . delta`` at unit norm, whose solution is the
    minimum-norm point of the convex hull of the differences. The margin it
    achieves comes back with it, in logits per unit of activation norm: a margin
    at or below zero means ``v_k`` lies inside the cone of the others and the
    class cannot be written by *any* additive edit — which is worth crashing on
    rather than steering past.
    """
    effective = (w[:-1] / std[:, None]).T

    directions, margins = [], []
    for index in range(len(effective)):
        direction, margin = _max_margin_direction(effective[index] - np.delete(effective, index, axis=0))
        if margin <= 1e-9:
            raise ValueError(
                f"class {index} cannot be written by any additive edit: its weight vector lies inside "
                f"the cone of the others, so some rival class wins at every magnitude"
            )
        directions.append(direction)
        margins.append(margin)
    return np.stack(directions), np.array(margins)


def class_write_accuracy(x: np.ndarray, w, mean, std, directions: np.ndarray, magnitude: float) -> float:
    """Fraction of cells where writing a class's direction makes the probe read it.

    The intervention's arithmetic, checked against the intervention's claim. The
    scalar-steering work had :func:`~goalmisgen.analysis.steering.verify` for
    exactly this reason: an off-by-one in the layer split produced a plausible
    number rather than a crash. Averaged over classes so a rare class cannot be
    hidden by a common one.
    """
    rates = []
    for index, direction in enumerate(directions):
        predicted = apply_multinomial(x + magnitude * direction, w, mean, std).argmax(1)
        rates.append(float((predicted == index).mean()))
    return float(np.mean(rates))
