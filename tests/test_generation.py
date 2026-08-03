"""Tests for maze layout generation.

Connectivity and acyclicity are checked with independent flood-fill / edge-count
implementations rather than by reusing the package's own solver, so a bug in the
solver cannot mask a bug in the generator.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from goalmisgen.envs.generation import RecursiveBacktracker, validate_shape

ODD_SHAPES = [(5, 5), (7, 7), (11, 9), (25, 25)]


def free_positions(walls: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(~walls)
    return set(zip(rows.tolist(), cols.tolist()))


def reachable_from(walls: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    """Flood fill over free cells, as an independent connectivity check."""
    height, width = walls.shape
    seen = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = (row + d_row, col + d_col)
            if not (0 <= neighbour[0] < height and 0 <= neighbour[1] < width):
                continue
            if walls[neighbour] or neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return seen


def count_free_adjacencies(walls: np.ndarray) -> int:
    """Number of edges in the graph of free cells."""
    horizontal = int(np.sum(~walls[:, :-1] & ~walls[:, 1:]))
    vertical = int(np.sum(~walls[:-1, :] & ~walls[1:, :]))
    return horizontal + vertical


@pytest.mark.parametrize("shape", [(4, 5), (5, 4), (10, 10)])
def test_rejects_even_dimensions(shape):
    with pytest.raises(ValueError, match="odd"):
        validate_shape(shape)


@pytest.mark.parametrize("shape", [(1, 5), (3, 1), (0, 0)])
def test_rejects_undersized_shapes(shape):
    with pytest.raises(ValueError, match="at least 3x3"):
        validate_shape(shape)


@pytest.mark.parametrize("shape", ODD_SHAPES)
def test_border_is_solid(shape):
    walls = RecursiveBacktracker().generate(shape, np.random.default_rng(0))
    assert walls[0, :].all()
    assert walls[-1, :].all()
    assert walls[:, 0].all()
    assert walls[:, -1].all()


@pytest.mark.parametrize("shape", ODD_SHAPES)
def test_every_cell_is_carved(shape):
    """All odd-index lattice positions must be free, i.e. no cell is stranded."""
    walls = RecursiveBacktracker().generate(shape, np.random.default_rng(1))
    height, width = shape
    for row in range(1, height - 1, 2):
        for col in range(1, width - 1, 2):
            assert not walls[row, col], f"cell {(row, col)} was never visited"


@pytest.mark.parametrize("shape", ODD_SHAPES)
def test_maze_is_fully_connected(shape):
    walls = RecursiveBacktracker().generate(shape, np.random.default_rng(2))
    free = free_positions(walls)
    assert reachable_from(walls, next(iter(free))) == free


@pytest.mark.parametrize("shape", ODD_SHAPES)
def test_unbraided_maze_is_a_tree(shape):
    """A perfect maze has exactly one path between any two cells."""
    walls = RecursiveBacktracker().generate(shape, np.random.default_rng(3))
    assert count_free_adjacencies(walls) == len(free_positions(walls)) - 1


def test_braiding_introduces_loops():
    shape = (25, 25)
    rng = np.random.default_rng(4)
    perfect = RecursiveBacktracker().generate(shape, np.random.default_rng(4))
    braided = RecursiveBacktracker(braid_probability=1.0).generate(shape, rng)

    assert count_free_adjacencies(perfect) == len(free_positions(perfect)) - 1
    assert count_free_adjacencies(braided) > len(free_positions(braided)) - 1
    # Loops must not come at the cost of connectivity.
    free = free_positions(braided)
    assert reachable_from(braided, next(iter(free))) == free


def test_braiding_keeps_the_border_solid():
    walls = RecursiveBacktracker(braid_probability=1.0).generate((15, 15), np.random.default_rng(5))
    assert walls[0, :].all() and walls[-1, :].all()
    assert walls[:, 0].all() and walls[:, -1].all()


@pytest.mark.parametrize("braid_probability", [0.0, 0.5])
def test_generation_is_deterministic_given_a_seed(braid_probability):
    generator = RecursiveBacktracker(braid_probability=braid_probability)
    first = generator.generate((15, 15), np.random.default_rng(7))
    second = generator.generate((15, 15), np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_different_seeds_give_different_mazes():
    generator = RecursiveBacktracker()
    mazes = [generator.generate((15, 15), np.random.default_rng(seed)) for seed in range(4)]
    assert not all(np.array_equal(mazes[0], other) for other in mazes[1:])


def test_rejects_invalid_braid_probability():
    with pytest.raises(ValueError, match="braid_probability"):
        RecursiveBacktracker(braid_probability=1.5)
