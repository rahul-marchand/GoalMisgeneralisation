"""Data model for a single maze episode.

A :class:`Level` is a complete, immutable description of one episode: the maze
layout, where the agent starts, and the objectives it may navigate to. It is
plain data — it knows nothing about rewards, observations, or how it was
sampled. Ground truth about the level (shortest paths, which objective is
optimal) is derived separately by :mod:`goalmisgen.envs.solver`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

Position = tuple[int, int]


@dataclasses.dataclass(frozen=True)
class Objective:
    """A goal the agent can navigate to, terminating the episode.

    The separation between ``value`` and ``feature_id`` is the mechanism by
    which goal misgeneralisation is studied. ``value`` is what the agent is
    rewarded for; ``feature_id`` is a surface property (rendered as colour)
    that may be correlated with value during training and decorrelated at test
    time. An agent that has learned to pursue ``feature_id`` rather than
    ``value`` will misgeneralise once the correlation is broken.
    """

    position: Position
    value: float
    feature_id: int

    def __post_init__(self) -> None:
        if self.feature_id < 0:
            raise ValueError(f"feature_id must be non-negative, got {self.feature_id}")


@dataclasses.dataclass(frozen=True, eq=False)
class Level:
    """An immutable maze episode specification.

    ``eq=False`` because the class holds a numpy array, for which the
    dataclass-generated ``__eq__`` would return an array rather than a bool.
    """

    walls: np.ndarray
    """Boolean array of shape ``(height, width)``. ``True`` marks a wall."""

    agent_start: Position
    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        walls = self.walls
        if walls.ndim != 2:
            raise ValueError(f"walls must be 2-dimensional, got shape {walls.shape}")
        if walls.dtype != np.bool_:
            raise TypeError(f"walls must have dtype bool, got {walls.dtype}")

        # Defensive copy, made read-only, so a Level cannot be mutated through a
        # reference the caller kept.
        frozen_walls = walls.copy()
        frozen_walls.flags.writeable = False
        object.__setattr__(self, "walls", frozen_walls)

        object.__setattr__(self, "objectives", tuple(self.objectives))
        if not self.objectives:
            raise ValueError("a Level must have at least one objective")

        self._validate_position(self.agent_start, "agent_start")
        for index, objective in enumerate(self.objectives):
            self._validate_position(objective.position, f"objectives[{index}].position")

        occupied = [self.agent_start, *(objective.position for objective in self.objectives)]
        if len(set(occupied)) != len(occupied):
            raise ValueError(
                "agent start and objective positions must all be distinct, got "
                f"agent_start={self.agent_start}, "
                f"objectives={[objective.position for objective in self.objectives]}"
            )

    def _validate_position(self, position: Position, name: str) -> None:
        row, col = position
        height, width = self.walls.shape
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{name}={position} is outside the {height}x{width} maze")
        if self.walls[row, col]:
            raise ValueError(f"{name}={position} is inside a wall")

    @property
    def shape(self) -> tuple[int, int]:
        return self.walls.shape  # type: ignore[return-value]

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    def is_wall(self, position: Position) -> bool:
        return bool(self.walls[position])

    def free_positions(self) -> list[Position]:
        """Every non-wall cell, in row-major order."""
        rows, cols = np.nonzero(~self.walls)
        return list(zip(rows.tolist(), cols.tolist()))
