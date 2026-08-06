"""Level ground truth recovered from an observation array.

The solver in :mod:`goalmisgen.envs.solver` computes distances from a
:class:`~goalmisgen.envs.level.Level`. Analysis code does not have one — a
rollout keeps the encoded observation and nothing else — so the same facts have
to be read back out of the channel stack. Everything here is the
observation-space twin of something in ``solver``, and
``tests/test_geometry.py`` asserts the two agree cell for cell on real levels.

Every array here is ``(height, width, channel)`` — the encoder's own order.
cleanba's wrapper transposes the *batched* observation to NCHW, so anything
arriving from a live rollout has already been moved back by
:func:`~goalmisgen.analysis.activations.collect_rollouts`.

Two decisions here are load-bearing, both of which exist to make a mistake
crash rather than produce a plausible number.

**Unreachable cells are ``NaN``, never ``-1``.** ``solver.distance_field`` uses
``-1`` as its sentinel, which is correct for an integer field but catastrophic
as a probe label: on 11x11 levels roughly one free cell in six is unreachable to
a given objective once the other blocks the route, and a distance field whose
mean is 12 does not notice ``-1`` values being regressed onto. It reads as a
slightly worse probe, not as an error.

**The channel layout is checked, not inferred.** Five channels is
``2 + 2 features + 1 value`` under the ``at_objective`` encoding, but it is also
``2 + 1 feature + 2 values`` under ``broadcast``. Guessing wrong would silently
probe the value channel as though it marked an objective, so ``n_features`` is
passed explicitly and :func:`check_layout` verifies it.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.envs.level import Position
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL
from goalmisgen.envs.solver import UNREACHABLE, distance_field


def check_layout(observation: np.ndarray, n_features: int) -> None:
    """Raise unless the observation looks like ``n_features`` objective channels.

    Each feature channel must mark exactly one cell, and that cell must not be a
    wall. A wrong ``n_features`` otherwise passes silently and probes whichever
    channel happens to sit at that index.
    """
    if observation.ndim != 3:
        raise ValueError(f"expected a (height, width, channel) observation, got shape {observation.shape}")
    needed = FIRST_FEATURE_CHANNEL + n_features
    if observation.shape[2] < needed:
        raise ValueError(
            f"n_features={n_features} needs at least {needed} channels but the observation has {observation.shape[2]}"
        )

    walls = observation[:, :, WALL_CHANNEL] > 0.5
    seen: list[Position] = []
    for feature in range(n_features):
        channel = observation[:, :, FIRST_FEATURE_CHANNEL + feature]
        # An objective channel is an indicator. A value channel carries the
        # objective's worth, so it holds something other than 0 and 1 — which is
        # what distinguishes "2 features + 1 value" from "1 feature + 2 values"
        # when both come to five channels. Counting marked cells does not: a
        # value channel holding 1.0 and 0.5 also marks exactly one cell at a
        # 0.5 threshold.
        if not np.isin(channel, (0.0, 1.0)).all():
            raise ValueError(
                f"channel {FIRST_FEATURE_CHANNEL + feature} is not a 0/1 indicator, so it is not an objective "
                f"channel; n_features={n_features} is wrong for this encoder"
            )
        marked = np.argwhere(channel > 0.5)
        if len(marked) != 1:
            raise ValueError(
                f"channel {FIRST_FEATURE_CHANNEL + feature} marks {len(marked)} cells, expected exactly 1; "
                f"n_features={n_features} is probably wrong for this encoder"
            )
        cell = (int(marked[0][0]), int(marked[0][1]))
        if walls[cell]:
            raise ValueError(f"objective for feature {feature} sits inside a wall at {cell}")
        if cell in seen:
            raise ValueError(f"two feature channels both mark {cell}")
        seen.append(cell)


def free_cells(observation: np.ndarray) -> np.ndarray:
    """Boolean mask of cells the agent could stand on. Padding counts as wall."""
    return observation[:, :, WALL_CHANNEL] < 0.5


def agent_cell(observation: np.ndarray) -> Position:
    marked = np.argwhere(observation[:, :, AGENT_CHANNEL] > 0.5)
    if len(marked) != 1:
        raise ValueError(f"expected exactly one agent cell, found {len(marked)}")
    return int(marked[0][0]), int(marked[0][1])


def objective_cell(observation: np.ndarray, feature_id: int) -> Position:
    marked = np.argwhere(observation[:, :, FIRST_FEATURE_CHANNEL + feature_id] > 0.5)
    if len(marked) != 1:
        raise ValueError(f"expected exactly one objective for feature {feature_id}, found {len(marked)}")
    return int(marked[0][0]), int(marked[0][1])


def blocking_walls(observation: np.ndarray, feature_id: int, n_features: int) -> np.ndarray:
    """Walls, plus every objective except ``feature_id``, marked impassable.

    Reaching any objective ends the episode, so a route crossing a different one
    never arrives. The twin of ``solver.walls_blocking_other_objectives``.
    """
    walls = observation[:, :, WALL_CHANNEL] > 0.5
    for other in range(n_features):
        if other != feature_id:
            walls[objective_cell(observation, other)] = True
    return walls


def bfs_field(walls: np.ndarray, source: Position) -> np.ndarray:
    """Shortest-path distance from ``source`` to every cell, ``NaN`` if unreachable.

    A float field rather than ``solver.distance_field``'s integers, precisely so
    that unreachable cells cannot be regressed onto as the number -1.
    """
    raw = distance_field(walls, source)
    field = raw.astype(np.float64)
    field[raw == UNREACHABLE] = np.nan
    return field


def manhattan_field(shape: tuple[int, int], source: Position) -> np.ndarray:
    """``|dr| + |dc|`` from ``source``. The free-space geodesic for 4-way moves.

    This is the null the distance-field claim has to beat: it needs no maze
    solving whatever, and it is a lower bound on the true distance everywhere.
    """
    rows, cols = np.indices(shape)
    return (np.abs(rows - source[0]) + np.abs(cols - source[1])).astype(np.float64)


def chebyshev_field(shape: tuple[int, int], source: Position) -> np.ndarray:
    """``max(|dr|, |dc|)`` from ``source`` — how far a 3x3 kernel has propagated.

    Weaker than :func:`manhattan_field` as a null, since the agent cannot move
    diagonally, but it is the metric in which an untrained convolutional tower
    spreads information, so it is the shape a probe would find if it were
    reading receptive-field growth rather than a computed distance.
    """
    rows, cols = np.indices(shape)
    return np.maximum(np.abs(rows - source[0]), np.abs(cols - source[1])).astype(np.float64)
