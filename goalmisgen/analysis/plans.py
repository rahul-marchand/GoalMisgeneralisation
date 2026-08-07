"""The route as a per-cell *directional* concept, and plans written by hand.

The distance field asks "how far is this cell". This asks the question the
planning-interpretability work asks: **which way does the route go from here**,
as one of five classes per cell — the four moves, or ``NEVER`` for a cell the
route does not use.

The difference matters for intervention rather than for decoding. A scalar
field can only be nudged: setting one cell's distance to 3 when its neighbours
say 8 describes no maze, so the network is handed a contradiction and
(reasonably) ignores it. A directional plan can be *rewritten* — write RIGHT
here, DOWN there, NEVER along the old route, and the edit is a coherent
alternative plan rather than a corrupted copy of the current one.

Two grids, and keeping them apart is the whole design:

:func:`observed_directions` reads the route the agent actually walked. It is the
probe's **label** — what the network is asked to have represented.

:func:`planned_directions` computes a route by breadth-first search to a chosen
objective, whether or not the agent went there. It is the probe's **output**,
written back into the activations. Fitting on one and intervening with the other
is what makes the intervention a hypothesis about the representation rather than
a replay of the behaviour.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.analysis import geometry
from goalmisgen.envs.level import Position
from goalmisgen.envs.solver import MOVES, shortest_path

NEVER = len(MOVES)
"""Class for a cell the route does not pass through. Four moves, then this."""

N_CLASSES = NEVER + 1

CLASS_NAMES: tuple[str, ...] = ("up", "down", "left", "right", "never")

UNSCOREABLE = -1
"""Marks a cell that must not be fitted or scored — a wall, or the route's end.

Not ``NEVER``: the last cell of a route has no next move, so calling it NEVER
would teach the probe that the objective is off-plan, which is the opposite of
true. ``NaN`` is unavailable because these labels are integer classes.
"""


def _direction(source: Position, target: Position) -> int | None:
    """Which move takes ``source`` to ``target``, or ``None`` if none does."""
    step = (target[0] - source[0], target[1] - source[1])
    return MOVES.index(step) if step in MOVES else None


def observed_directions(rollout) -> np.ndarray:
    """``(height, width)`` class per cell, from the route the agent walked.

    Reconstructed by ordering the visited cells by arrival step rather than by
    looking for a cell arriving at ``step + 1``. The agent can walk into a wall,
    which advances the step counter without moving it, so arrival steps have
    gaps in them; their *order* is still the route.
    """
    labels = np.full(rollout.visited.shape, UNSCOREABLE, dtype=np.int64)

    visited = np.argwhere(rollout.visit_step >= 0)
    if len(visited) < 2:
        return labels
    order = visited[np.argsort(rollout.visit_step[tuple(visited.T)], kind="stable")]

    for current, following in zip(order, order[1:]):
        step = _direction((int(current[0]), int(current[1])), (int(following[0]), int(following[1])))
        if step is not None:
            labels[current[0], current[1]] = step

    # Everything free and off-route is a genuine NEVER. The final cell keeps
    # UNSCOREABLE: it is on the route but has no next move.
    off_route = geometry.free_cells(rollout.observation) & (rollout.visit_step < 0)
    labels[off_route] = NEVER
    return labels


def planned_directions(observation: np.ndarray, feature_id: int, n_features: int = 2) -> np.ndarray | None:
    """The route a shortest-path planner would take to one objective.

    ``None`` if that objective cannot be reached — the other one blocks the only
    corridor, which happens often enough on 11x11 levels to matter.

    Off-route free cells are ``NEVER`` so the returned grid is a *complete*
    plan: writing it into the activations says both where the route goes and
    where it does not, which is the pair of edits the intervention needs.
    """
    walls = geometry.blocking_walls(observation, feature_id, n_features)
    path = shortest_path(walls, geometry.agent_cell(observation), geometry.objective_cell(observation, feature_id))
    if path is None:
        return None

    labels = np.full(observation.shape[:2], UNSCOREABLE, dtype=np.int64)
    labels[geometry.free_cells(observation)] = NEVER
    for current, following in zip(path, path[1:]):
        step = _direction(current, following)
        if step is None:
            raise RuntimeError(f"path step {current} -> {following} is not a legal move")
        labels[current] = step
    labels[path[-1]] = UNSCOREABLE
    return labels


def class_counts(labels: np.ndarray) -> dict[str, int]:
    """How many cells fall in each class, for reporting the class balance."""
    return {name: int((labels == index).sum()) for index, name in enumerate(CLASS_NAMES)}
