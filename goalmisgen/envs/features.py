"""How surface features are attached to objectives — the proxy under study.

``feature_id`` is a property of an objective that carries no reward information
of its own but which training may make *predictive* of reward. An agent that
learns to pursue the feature rather than the value will generalise badly the
moment that relationship is broken. Everything about that relationship lives
here, behind a protocol, because varying it is the project's central
manipulation and each new experiment should be a new class rather than an edit.

A scheme sees only the objectives' *values*. A proxy correlated with something
else — distance, say — would need that signature broadened to carry the extra
quantity; the sampler has distances to hand, so it is a small change, but it is
a real one rather than just a new class.

The chance level is ``1 / n_objectives``, not one half. With two objectives a
correlation of 0.5 means colour is uninformative; with three it is already
better than chance and is *not* a control. :meth:`FeatureScheme.chance` makes
that explicit so evaluation conditions can be built relative to it.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np


class FeatureScheme(Protocol):
    """Assigns a feature id to each objective, given their values."""

    def assign(self, values: tuple[float, ...], rng: np.random.Generator) -> tuple[int, ...]:
        ...

    def canonical(self) -> "FeatureScheme":
        """An equivalent scheme with the proxy relationship neutralised.

        Used when pre-generating levels: assignment consumes random draws, so a
        dataset must be built with one fixed scheme or its *layouts* would
        depend on the correlation. Returning a scheme that draws identically
        keeps stored content independent of the correlation actually trained on.
        """
        ...

    def chance(self, n_objectives: int) -> float:
        """The correlation a scheme carrying no information would show."""
        ...


@dataclasses.dataclass(frozen=True)
class CorrelatedFeatures:
    """Feature 0 marks the highest-value objective with probability ``correlation``.

    Remaining features go to the remaining objectives uniformly at random, so no
    feature other than 0 carries information about value.
    """

    correlation: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.correlation <= 1.0:
            raise ValueError(f"correlation must be in [0, 1], got {self.correlation}")

    def chance(self, n_objectives: int) -> float:
        return 1.0 / n_objectives

    def canonical(self) -> "CorrelatedFeatures":
        return dataclasses.replace(self, correlation=1.0)

    def assign(self, values: tuple[float, ...], rng: np.random.Generator) -> tuple[int, ...]:
        n = len(values)
        if n == 1:
            return (0,)

        # Break ties at random. np.argmax always returns the lowest index, so
        # with values like (1.0, 1.0, 0.5) feature 0 would land on objective 0
        # every time and the correlation would control nothing, while every
        # metric continued to report normally.
        maxima = np.flatnonzero(np.asarray(values) >= max(values) - 1e-12)
        best = int(maxima[rng.integers(len(maxima))])

        # Both draws happen whichever branch is taken. Drawing the alternative
        # only when it is used made the number of random values consumed depend
        # on the correlation, so with three or more objectives the level streams
        # of two correlation arms diverged after the first episode - scoring
        # them on different mazes, which is what the shared seed exists to stop.
        follows_value = rng.random() < self.correlation
        others = [index for index in range(n) if index != best]
        alternative = others[int(rng.integers(len(others)))]
        holder_of_feature_zero = best if follows_value else alternative

        remaining = [index for index in range(n) if index != holder_of_feature_zero]
        rng.shuffle(remaining)

        feature_ids = [0] * n
        for feature_id, objective_index in enumerate(remaining, start=1):
            feature_ids[objective_index] = feature_id
        return tuple(feature_ids)
