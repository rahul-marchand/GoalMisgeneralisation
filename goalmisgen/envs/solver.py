"""Ground truth for maze levels.

Everything the agent's behaviour is *measured against* is computed here:
shortest-path distances, the utility of each objective, and which objective an
optimal agent would choose. This module is the reference an agent can be wrong
relative to, so it is deliberately simple and separately tested.

Nothing here is visible to the agent. Results reach experiments through the
environment's ``info`` dict, never through the observation.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque

import numpy as np

from goalmisgen.envs.level import Level, Position

UNREACHABLE = -1
"""Sentinel in a distance field for cells not reachable from the source."""

TIE_TOLERANCE = 1e-9
"""Utilities within this of the maximum count as tied for optimal."""

# Fixed iteration order, so path reconstruction is deterministic when a maze
# contains loops and several shortest paths exist.
_MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def distance_field(walls: np.ndarray, source: Position) -> np.ndarray:
    """Breadth-first shortest-path distance from ``source`` to every free cell.

    Returns an int array shaped like ``walls``, holding :data:`UNREACHABLE` for
    walls and for free cells in a disconnected component.
    """
    height, width = walls.shape
    row, col = source
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"source {source} is outside the {height}x{width} maze")
    if walls[source]:
        raise ValueError(f"source {source} is inside a wall")

    distances = np.full(walls.shape, UNREACHABLE, dtype=np.int32)
    distances[source] = 0

    queue = deque([source])
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for d_row, d_col in _MOVES:
            neighbour = (current[0] + d_row, current[1] + d_col)
            if not (0 <= neighbour[0] < height and 0 <= neighbour[1] < width):
                continue
            if walls[neighbour] or distances[neighbour] != UNREACHABLE:
                continue
            distances[neighbour] = next_distance
            queue.append(neighbour)

    return distances


def shortest_path(walls: np.ndarray, source: Position, target: Position) -> tuple[Position, ...] | None:
    """One shortest path from ``source`` to ``target``, inclusive of both.

    Returns ``None`` if ``target`` is unreachable. In a perfect maze the path is
    unique; if the layout has loops this returns a deterministic choice among
    the equally short ones.
    """
    distances = distance_field(walls, source)
    if distances[target] == UNREACHABLE:
        return None

    height, width = walls.shape
    path = [target]
    current = target
    while current != source:
        wanted = distances[current] - 1
        for d_row, d_col in _MOVES:
            neighbour = (current[0] + d_row, current[1] + d_col)
            if not (0 <= neighbour[0] < height and 0 <= neighbour[1] < width):
                continue
            if distances[neighbour] == wanted:
                current = neighbour
                break
        else:  # pragma: no cover - unreachable given a valid distance field
            raise RuntimeError(f"no descending neighbour from {current}; distance field is inconsistent")
        path.append(current)

    return tuple(reversed(path))


@dataclasses.dataclass(frozen=True)
class LevelSolution:
    """What an optimal agent would do on a level, and by how much.

    Distances and utilities are ``None`` for objectives that cannot be reached.
    """

    distances: tuple[int | None, ...]
    utilities: tuple[float | None, ...]
    optimal_index: int
    """Best objective by utility. Ties are broken by lowest index."""

    optimal_indices: tuple[int, ...]
    """Every objective tied for best, so ambiguous levels can be filtered out."""

    utility_margin: float
    """Utility gap between the best objective and the best alternative.

    ``inf`` when only one objective is reachable. Small margins mark levels near
    the decision boundary, where even an optimal-in-expectation agent will look
    inconsistent; experiments should usually report these separately.
    """

    @property
    def is_ambiguous(self) -> bool:
        return len(self.optimal_indices) > 1


def walls_blocking_other_objectives(level: Level, index: int) -> np.ndarray:
    """``level.walls`` with every objective except ``index`` marked impassable.

    Reaching any objective ends the episode, so a route that crosses a different
    objective never arrives. Distances must therefore be measured on a maze in
    which the others are obstacles, or the "optimal" choice can be one the agent
    is physically unable to take.
    """
    walls = level.walls.copy()
    for other, objective in enumerate(level.objectives):
        if other != index:
            walls[objective.position] = True
    return walls


def path_to_objective(level: Level, index: int) -> tuple[Position, ...] | None:
    """Shortest route to objective ``index`` that avoids the other objectives.

    This is the trajectory an optimal agent aiming at ``index`` would take, and
    so the correct source of "which cells will the agent step on" probe labels.
    """
    return shortest_path(
        walls_blocking_other_objectives(level, index),
        level.agent_start,
        level.objectives[index].position,
    )


def objective_distances(level: Level) -> tuple[int | None, ...]:
    """Steps to each objective, routing around the others. ``None`` if blocked off."""
    distances: list[int | None] = []
    for index, objective in enumerate(level.objectives):
        walls = walls_blocking_other_objectives(level, index)
        raw = int(distance_field(walls, level.agent_start)[objective.position])
        distances.append(None if raw == UNREACHABLE else raw)
    return tuple(distances)


def solve(level: Level, step_penalty: float) -> LevelSolution:
    """Evaluate every objective as ``value - step_penalty * distance``.

    ``step_penalty`` is an environment parameter rather than a property of the
    level, which is why the answer is computed here instead of being stored on
    :class:`~goalmisgen.envs.level.Level`.

    Distances route around the other objectives; see
    :func:`walls_blocking_other_objectives`.
    """
    if step_penalty < 0:
        raise ValueError(f"step_penalty must be non-negative, got {step_penalty}")

    distances = objective_distances(level)
    utilities: list[float | None] = [
        None if distance is None else objective.value - step_penalty * distance
        for objective, distance in zip(level.objectives, distances)
    ]

    reachable = [(index, utility) for index, utility in enumerate(utilities) if utility is not None]
    if not reachable:
        raise ValueError("no objective is reachable from the agent's start position")

    best_utility = max(utility for _, utility in reachable)
    optimal_indices = tuple(index for index, utility in reachable if abs(utility - best_utility) <= TIE_TOLERANCE)

    alternatives = [utility for index, utility in reachable if index not in optimal_indices]
    utility_margin = best_utility - max(alternatives) if alternatives else math.inf

    return LevelSolution(
        distances=distances,
        utilities=tuple(utilities),
        optimal_index=optimal_indices[0],
        optimal_indices=optimal_indices,
        utility_margin=utility_margin,
    )
