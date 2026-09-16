"""One canonical route rule, so every expert decision is a function of the map.

An imitation target must be *consistent*: two fields that look the same must
get the same route, and the rule that picks it must be readable from the
observation by a model that emits the route forwards, one move at a time.
The maze solver's ``shortest_path`` is deterministic but breaks ties by
walking backwards from the target in a fixed move order, which a forward
model cannot see; on braided open fields, where shortest paths are many, the
stage-2 model imitating it lost most of its routes to illegal moves, and the
stage-4 planner's tree order (ranked over permutations of static distances)
was worse still: not inferable at all.

The rule here: **at each step take the first move, in a fixed order, that
shortens the distance to the target.** Distances come from one breadth-first
search from the target, so the route is a shortest path; the tie-break is a
constant preference (up, down, left, right) the model can learn once.
:func:`nearest` applies the same idea to choosing *which* of several cells to
go to: the closest, ties by row-major order.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from goalmisgen.envs.level import Position
from goalmisgen.envs.solver import MOVES, UNREACHABLE, distance_field

ORDER: tuple[int, ...] = (0, 1, 2, 3)
"""Move preference for tie-breaks, as indices into ``MOVES``: up, down, left, right."""


def distances_to(walls: np.ndarray, target: Position) -> np.ndarray:
    """Distance from every free cell to ``target``, which is opened if it is solid."""
    grid = np.asarray(walls, dtype=np.bool_).copy()
    grid[target] = False
    return distance_field(grid, target)


def canonical_moves(walls: np.ndarray, source: Position, target: Position, order: Sequence[int] = ORDER) -> list[int] | None:
    """Maze-move indices from ``source`` onto ``target`` along a shortest path, canonically tie-broken.

    The last move steps onto the target cell, which may be solid (an ore, a
    tree): in the engine that move is the turn to face it. ``None`` if no
    route exists. ``source == target`` gives an empty route.
    """
    field = distances_to(walls, target)
    height, width = field.shape
    if field[source] == UNREACHABLE:
        return None
    moves: list[int] = []
    here = source
    while here != target:
        wanted = int(field[here]) - 1
        for move in order:
            d_row, d_col = MOVES[move]
            nxt = (here[0] + d_row, here[1] + d_col)
            if 0 <= nxt[0] < height and 0 <= nxt[1] < width and field[nxt] == wanted:
                moves.append(move)
                here = nxt
                break
        else:  # pragma: no cover - a consistent distance field always has a descending neighbour
            raise RuntimeError("no descending neighbour; the distance field is inconsistent")
    return moves


def nearest(walls: np.ndarray, source: Position, cells: Sequence[Position]) -> tuple[Position, int] | None:
    """The cell in ``cells`` with the shortest route from ``source``, ties by row-major order.

    Solid cells count as reachable when a neighbour is; the distance is the
    number of moves of :func:`canonical_moves`, which ends on the cell.
    """
    field = distance_field(np.asarray(walls, dtype=np.bool_), source)
    height, width = field.shape
    best: tuple[int, Position] | None = None
    for cell in sorted(cells):
        if not walls[cell]:
            d = int(field[cell])
            if d == UNREACHABLE:
                continue
        else:
            options = [
                int(field[cell[0] + dr, cell[1] + dc]) + 1
                for dr, dc in MOVES
                if 0 <= cell[0] + dr < height
                and 0 <= cell[1] + dc < width
                and field[cell[0] + dr, cell[1] + dc] != UNREACHABLE
            ]
            if not options:
                continue
            d = min(options)
        if best is None or d < best[0]:
            best = (d, cell)
    return None if best is None else (best[1], best[0])
