"""Behavioural flip points: the model's own gap reading, measured without probes.

The value-axis grid provides one computation with its threshold dialled: many
models, each with a measured crossing theta, all decoded on the same levels.
Under the utility rule ``c = 1[gap * (1 + eps) < G]`` a level is taken exactly
by the models whose threshold exceeds its effective gap, so as models are
ordered by theta the level's choices form a step - and the step's location
*is* ``gap * (1 + eps)``, the model's own reading of that level's gap in
steps. No activations are involved, which is what makes this the validation
target for probes: a probe reads the model's estimate to the extent it
predicts the flip point better than the true gap does.

The fit is a per-level step function, not a logistic - the method note in
UtilityRule.md applies here too.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class FlipPoints:
    """Per-level flip thresholds over a family of theta-ordered models."""

    thetas: np.ndarray
    """(M,) - each model's measured crossing, ascending."""

    flip: np.ndarray
    """(N,) float - the theta at which the level flips to taken; NaN if censored."""

    violations: np.ndarray
    """(N,) int - choices the best step function cannot explain (0 = a clean step)."""

    censored_low: np.ndarray
    """(N,) bool - taken by every model, so the flip lies below the grid."""

    censored_high: np.ndarray
    """(N,) bool - taken by none, so the flip lies above the grid."""

    @property
    def bracketed(self) -> np.ndarray:
        return np.isfinite(self.flip)


def step_fit(thetas: np.ndarray, chosen: np.ndarray) -> FlipPoints:
    """Fit one step per level to choices across theta-ordered models.

    ``chosen`` is ``(M, N)`` bool: did model ``m`` take the richer objective on
    level ``i``. Rows must already be sorted by ascending theta. For each level
    the split index ``k`` minimising misclassifications - taken below the step
    plus not-taken at or above it - places the flip between ``thetas[k - 1]``
    and ``thetas[k]``; the midpoint is reported. Ties take the smallest ``k``,
    a half-grid-step conservative bias, small next to the grid spacing.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    chosen = np.asarray(chosen)
    if chosen.ndim != 2 or len(thetas) != len(chosen):
        raise ValueError(f"chosen must be (M, N) with M == len(thetas); got {chosen.shape} against {len(thetas)}")
    if np.any(np.diff(thetas) < 0):
        raise ValueError("thetas must be ascending; sort the models first")

    M, N = chosen.shape
    C = chosen.astype(np.int64)
    taken_below = np.vstack([np.zeros((1, N), np.int64), np.cumsum(C, axis=0)])  # (M+1, N)
    total = taken_below[-1]
    # cost(k) = taken below the step + not-taken at or above it
    cost = taken_below + ((M - np.arange(M + 1))[:, None] - (total[None, :] - taken_below))
    k = np.argmin(cost, axis=0)
    violations = cost[k, np.arange(N)]

    censored_low, censored_high = k == 0, k == M
    flip = np.full(N, np.nan)
    mid = ~censored_low & ~censored_high
    flip[mid] = (thetas[k[mid] - 1] + thetas[k[mid]]) / 2
    return FlipPoints(
        thetas=thetas,
        flip=flip,
        violations=violations.astype(np.int64),
        censored_low=censored_low,
        censored_high=censored_high,
    )
