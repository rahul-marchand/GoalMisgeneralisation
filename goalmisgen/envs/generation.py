"""Maze layout generation.

Layouts live on an odd-sized grid: cells sit at odd indices and the even
indices between them are potential walls. A maze of ``(2n+1, 2m+1)`` therefore
holds ``n x m`` cells surrounded by a solid border.

:class:`MazeGenerator` is the extension point. Adding a different layout
distribution (Kruskal, Prim, rooms-and-corridors, ...) means adding a class
here, not modifying the environment.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

# Offsets to the four neighbouring *cells*, which are two grid steps apart.
_CELL_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = ((-2, 0), (2, 0), (0, -2), (0, 2))


class MazeGenerator(Protocol):
    """Produces a wall grid. ``True`` marks a wall."""

    def generate(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray: ...


def validate_shape(shape: tuple[int, int]) -> None:
    """Reject shapes that cannot hold a cell lattice with a surrounding border."""
    height, width = shape
    if height < 3 or width < 3:
        raise ValueError(f"maze shape must be at least 3x3, got {height}x{width}")
    if height % 2 == 0 or width % 2 == 0:
        raise ValueError(
            f"maze dimensions must be odd so that cells sit at odd indices, got {height}x{width}"
        )


@dataclasses.dataclass(frozen=True)
class RecursiveBacktracker:
    """Randomised depth-first search, producing a uniformly connected maze.

    With ``braid_probability == 0`` the result is a *perfect* maze: exactly one
    path between any two cells. Perfect mazes make the shortest path unique,
    which keeps probe labels ("which cells will the agent step on") unambiguous.

    ``braid_probability`` optionally removes dead ends, introducing loops and so
    multiple routes. Kept available because unique-path mazes may make the
    planning problem unrepresentatively easy, but off by default.
    """

    braid_probability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.braid_probability <= 1.0:
            raise ValueError(f"braid_probability must be in [0, 1], got {self.braid_probability}")

    def generate(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        validate_shape(shape)
        height, width = shape

        walls = np.ones(shape, dtype=np.bool_)

        start = (
            int(rng.integers((height - 1) // 2) * 2 + 1),
            int(rng.integers((width - 1) // 2) * 2 + 1),
        )
        walls[start] = False

        # Explicit stack rather than recursion: a 25x25 maze would otherwise
        # nest ~150 frames deep, and larger evaluation mazes far more.
        stack: list[tuple[int, int]] = [start]
        while stack:
            row, col = stack[-1]
            unvisited = [
                (row + d_row, col + d_col)
                for d_row, d_col in _CELL_NEIGHBOUR_OFFSETS
                if 0 < row + d_row < height - 1 and 0 < col + d_col < width - 1 and walls[row + d_row, col + d_col]
            ]
            if not unvisited:
                stack.pop()
                continue

            next_row, next_col = unvisited[int(rng.integers(len(unvisited)))]
            walls[(row + next_row) // 2, (col + next_col) // 2] = False
            walls[next_row, next_col] = False
            stack.append((next_row, next_col))

        if self.braid_probability > 0.0:
            self._braid(walls, rng)

        return walls

    def _braid(self, walls: np.ndarray, rng: np.random.Generator) -> None:
        """Open a wall at randomly chosen dead ends, creating loops."""
        height, width = walls.shape
        for row in range(1, height - 1, 2):
            for col in range(1, width - 1, 2):
                if rng.random() >= self.braid_probability:
                    continue

                open_neighbours = 0
                closed: list[tuple[int, int]] = []
                for d_row, d_col in _CELL_NEIGHBOUR_OFFSETS:
                    next_row, next_col = row + d_row, col + d_col
                    if not (0 < next_row < height - 1 and 0 < next_col < width - 1):
                        continue
                    if walls[(row + next_row) // 2, (col + next_col) // 2]:
                        closed.append(((row + next_row) // 2, (col + next_col) // 2))
                    else:
                        open_neighbours += 1

                is_dead_end = open_neighbours <= 1
                if is_dead_end and closed:
                    walls[closed[int(rng.integers(len(closed)))]] = False
