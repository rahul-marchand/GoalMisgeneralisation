"""Tests for maze ground truth.

Distances are checked against scipy's Dijkstra on an explicitly constructed
adjacency graph — an implementation with nothing in common with our BFS — so
the two cannot share a bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from goalmisgen.envs.generation import RecursiveBacktracker
from goalmisgen.envs.level import Level, Objective
from goalmisgen.envs.solver import (
    UNREACHABLE,
    distance_field,
    path_to_objective,
    shortest_path,
    solve,
    walls_blocking_other_objectives,
)

OPEN_ROOM = np.ones((5, 5), dtype=np.bool_)
OPEN_ROOM[1:4, 1:4] = False


def reference_distances(walls: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Shortest-path distances via scipy, as an independent reference."""
    height, width = walls.shape
    n_nodes = height * width
    rows, cols = [], []
    for row in range(height):
        for col in range(width):
            if walls[row, col]:
                continue
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < height and 0 <= n_col < width and not walls[n_row, n_col]:
                    rows.append(row * width + col)
                    cols.append(n_row * width + n_col)

    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
    flat = dijkstra(graph, indices=source[0] * width + source[1])
    result = np.where(np.isinf(flat), UNREACHABLE, flat).astype(np.int32).reshape(height, width)
    result[walls] = UNREACHABLE
    return result


@pytest.mark.parametrize("seed", range(8))
def test_distance_field_matches_scipy_on_random_mazes(seed):
    walls = RecursiveBacktracker(braid_probability=0.3).generate((15, 15), np.random.default_rng(seed))
    source = (1, 1)
    assert np.array_equal(distance_field(walls, source), reference_distances(walls, source))


def test_distance_field_on_an_open_room():
    distances = distance_field(OPEN_ROOM, (1, 1))
    assert distances[1, 1] == 0
    assert distances[1, 2] == 1
    assert distances[3, 3] == 4  # Manhattan, no obstacles in between
    assert distances[0, 0] == UNREACHABLE  # wall


def test_distance_field_marks_disconnected_regions_unreachable():
    walls = np.ones((5, 7), dtype=np.bool_)
    walls[1:4, 1:2] = False  # left corridor
    walls[1:4, 4:5] = False  # right corridor, no connection
    distances = distance_field(walls, (1, 1))
    assert distances[3, 1] == 2
    assert distances[1, 4] == UNREACHABLE


def test_distance_field_rejects_a_source_in_a_wall():
    with pytest.raises(ValueError, match="inside a wall"):
        distance_field(OPEN_ROOM, (0, 0))


def test_distance_field_rejects_an_out_of_bounds_source():
    with pytest.raises(ValueError, match="outside"):
        distance_field(OPEN_ROOM, (99, 99))


@pytest.mark.parametrize("seed", range(8))
def test_shortest_path_is_contiguous_and_optimal(seed):
    walls = RecursiveBacktracker().generate((15, 15), np.random.default_rng(seed))
    source, target = (1, 1), (13, 13)
    path = shortest_path(walls, source, target)
    assert path is not None

    assert path[0] == source and path[-1] == target
    assert len(path) - 1 == distance_field(walls, source)[target]
    assert len(set(path)) == len(path), "path revisits a cell"
    for cell in path:
        assert not walls[cell], f"path passes through wall {cell}"
    for before, after in zip(path, path[1:]):
        assert abs(before[0] - after[0]) + abs(before[1] - after[1]) == 1, "path is not contiguous"


def test_shortest_path_to_an_unreachable_target_is_none():
    walls = np.ones((5, 7), dtype=np.bool_)
    walls[1:4, 1:2] = False
    walls[1:4, 4:5] = False
    assert shortest_path(walls, (1, 1), (1, 4)) is None


def test_shortest_path_to_the_source_is_a_single_cell():
    assert shortest_path(OPEN_ROOM, (1, 1), (1, 1)) == ((1, 1),)


def make_level(objectives) -> Level:
    return Level(walls=OPEN_ROOM, agent_start=(1, 1), objectives=objectives)


def test_solve_computes_distances_and_utilities():
    level = make_level(
        (
            Objective((1, 2), value=1.0, feature_id=0),
            Objective((3, 3), value=2.0, feature_id=1),
        )
    )
    solution = solve(level, step_penalty=0.1)
    assert solution.distances == (1, 4)
    assert solution.utilities == pytest.approx((0.9, 1.6))


def test_a_high_step_penalty_makes_the_near_objective_optimal():
    """The point of the step penalty: distance can outweigh value."""
    level = make_level(
        (
            Objective((1, 2), value=1.0, feature_id=0),  # distance 1
            Objective((3, 3), value=2.0, feature_id=1),  # distance 4
        )
    )
    assert solve(level, step_penalty=0.1).optimal_index == 1  # far but valuable wins
    assert solve(level, step_penalty=0.5).optimal_index == 0  # penalty flips it


def test_ties_are_reported():
    """Agent centred between two equally good objectives, plus a worse third."""
    level = Level(
        walls=OPEN_ROOM,
        agent_start=(2, 2),
        objectives=(
            Objective((1, 2), value=1.0, feature_id=0),  # distance 1 -> utility 0.5
            Objective((3, 2), value=1.0, feature_id=1),  # distance 1 -> utility 0.5
            Objective((2, 1), value=0.5, feature_id=2),  # distance 1 -> utility 0.0
        ),
    )
    solution = solve(level, step_penalty=0.5)
    assert solution.optimal_indices == (0, 1)
    assert solution.optimal_index == 0  # lowest index breaks the tie
    assert solution.is_ambiguous
    # A tie means no margin at all. Reporting the gap to the next *strictly*
    # worse objective would file exactly-ambiguous levels in the highest
    # confidence bin, where either choice scores as optimal.
    assert solution.utility_margin == pytest.approx(0.0)


def test_margin_is_infinite_with_a_single_objective():
    solution = solve(make_level((Objective((3, 3), value=1.0, feature_id=0),)), step_penalty=0.1)
    assert solution.utility_margin == math.inf
    assert not solution.is_ambiguous


def test_unreachable_objective_yields_none_and_is_never_optimal():
    walls = np.ones((5, 7), dtype=np.bool_)
    walls[1:4, 1:2] = False
    walls[1:4, 4:5] = False
    level = Level(
        walls=walls,
        agent_start=(1, 1),
        objectives=(
            Objective((3, 1), value=1.0, feature_id=0),
            Objective((1, 4), value=99.0, feature_id=1),  # walled off
        ),
    )
    solution = solve(level, step_penalty=0.1)
    assert solution.distances == (2, None)
    assert solution.utilities[1] is None
    assert solution.optimal_index == 0
    assert solution.utility_margin == math.inf


def test_solve_raises_when_nothing_is_reachable():
    walls = np.ones((5, 7), dtype=np.bool_)
    walls[1:4, 1:2] = False
    walls[1:4, 4:5] = False
    level = Level(
        walls=walls,
        agent_start=(1, 1),
        objectives=(Objective((1, 4), value=1.0, feature_id=0),),
    )
    with pytest.raises(ValueError, match="no objective is reachable"):
        solve(level, step_penalty=0.1)


def test_distance_routes_around_another_objective():
    """An objective on the direct route is an obstacle: stepping on it ends the episode.

    Corridor layout, agent at the left end, objective B in the middle, objective
    A at the right end. A cannot be reached in 4 steps because the episode would
    terminate at B after 2.
    """
    walls = np.ones((3, 7), dtype=np.bool_)
    walls[1, 1:6] = False
    level = Level(
        walls=walls,
        agent_start=(1, 1),
        objectives=(
            Objective((1, 5), value=1.0, feature_id=0),  # far end
            Objective((1, 3), value=0.5, feature_id=1),  # blocks the corridor
        ),
    )
    solution = solve(level, step_penalty=0.01)

    assert solution.distances[1] == 2  # the blocker is reachable
    assert solution.distances[0] is None  # the far objective is not
    assert solution.optimal_index == 1


def test_path_to_objective_avoids_the_other_objectives():
    walls = np.ones((5, 5), dtype=np.bool_)
    walls[1:4, 1:4] = False
    level = Level(
        walls=walls,
        agent_start=(1, 1),
        objectives=(
            Objective((1, 3), value=1.0, feature_id=0),
            Objective((1, 2), value=0.5, feature_id=1),  # directly in the way
        ),
    )
    path = path_to_objective(level, 0)
    assert path is not None
    assert (1, 2) not in path
    assert path[0] == (1, 1) and path[-1] == (1, 3)


def test_walls_blocking_other_objectives_leaves_the_target_free():
    level = make_level(
        (
            Objective((1, 2), value=1.0, feature_id=0),
            Objective((3, 3), value=0.5, feature_id=1),
        )
    )
    blocked = walls_blocking_other_objectives(level, 0)
    assert not blocked[1, 2], "the target itself must stay passable"
    assert blocked[3, 3], "the other objective must be blocked"
    assert not level.walls[3, 3], "the level itself must be unmodified"


def test_solve_rejects_negative_step_penalty():
    level = make_level((Objective((3, 3), value=1.0, feature_id=0),))
    with pytest.raises(ValueError, match="step_penalty"):
        solve(level, step_penalty=-0.1)


def test_zero_step_penalty_selects_purely_on_value():
    level = make_level(
        (
            Objective((1, 2), value=1.0, feature_id=0),
            Objective((3, 3), value=2.0, feature_id=1),
        )
    )
    assert solve(level, step_penalty=0.0).optimal_index == 1


def test_step_limit_excludes_objectives_that_cannot_be_reached_in_time():
    """A goal beyond the episode limit is not an achievable optimum.

    Without this the solver names an objective no policy can reach, putting a
    floor on chose_optimal that no agent can cross.
    """
    # An open room, not a corridor: in a one-wide corridor the near objective
    # would block the far one outright, which is a different effect.
    walls = np.ones((5, 21), dtype=np.bool_)
    walls[1:4, 1:20] = False
    level = Level(
        walls=walls,
        agent_start=(2, 1),
        objectives=(
            Objective((2, 19), value=1.0, feature_id=0),  # 18 steps away
            Objective((1, 3), value=0.6, feature_id=1),  # 3 steps away
        ),
    )
    generous = solve(level, step_penalty=0.01)
    assert generous.optimal_index == 0  # far but more valuable

    tight = solve(level, step_penalty=0.01, step_limit=10)
    assert tight.distances[0] is None, "unreachable in time must not count"
    assert tight.optimal_index == 1


def test_tied_maximum_values_do_not_pin_the_best_to_index_zero():
    """np.argmax always returns the lowest index, which would make the
    correlation knob silently control nothing when values tie."""
    from goalmisgen.envs.features import CorrelatedFeatures

    scheme = CorrelatedFeatures(correlation=1.0)
    rng = np.random.default_rng(0)
    holders = [scheme.assign((1.0, 1.0, 0.5), rng).index(0) for _ in range(400)]
    assert set(holders) == {0, 1}, "feature 0 should land on either tied maximum"
    share = holders.count(0) / len(holders)
    assert 0.35 < share < 0.65, f"tie-break is skewed: {share:.2f}"
