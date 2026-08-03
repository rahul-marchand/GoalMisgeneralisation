"""How objective values are drawn.

Kept separate from placement because the two vary independently across
experiments. Fixed values make the task a distance comparison; randomised
values force the agent to actually read and compare the values it is shown,
which is the variant that yields a genuine value-comparison mechanism rather
than a geometric heuristic.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np


class ValueScheme(Protocol):
    """Draws the reward value of each objective for one episode."""

    def sample(self, n_objectives: int, rng: np.random.Generator) -> tuple[float, ...]:
        ...


@dataclasses.dataclass(frozen=True)
class FixedValues:
    """The same values every episode.

    The optimal policy then reduces to a threshold on the difference in
    distances, which an agent can implement without representing value at all.
    Fine for a first agent; see :class:`UniformValues` for the harder variant.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("FixedValues needs at least one value")

    def sample(self, n_objectives: int, rng: np.random.Generator) -> tuple[float, ...]:
        del rng  # deterministic by construction
        if n_objectives != len(self.values):
            raise ValueError(f"FixedValues was given {len(self.values)} values but the sampler asked for {n_objectives}")
        return self.values


@dataclasses.dataclass(frozen=True)
class UniformValues:
    """Independent uniform draws, resampled every episode.

    Intended to be paired with value channels in the observation, so the agent
    must read the values to act optimally.
    """

    low: float = 0.25
    high: float = 1.0

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"require low < high, got low={self.low}, high={self.high}")

    def sample(self, n_objectives: int, rng: np.random.Generator) -> tuple[float, ...]:
        return tuple(float(value) for value in rng.uniform(self.low, self.high, size=n_objectives))
