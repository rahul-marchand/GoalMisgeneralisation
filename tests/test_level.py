"""Tests for the Level / Objective data model."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.level import Level, Objective

# A 5x5 maze with a plus-shaped open region:
#   # # # # #
#   # . . . #
#   # . . . #
#   # . . . #
#   # # # # #
OPEN_ROOM = np.ones((5, 5), dtype=np.bool_)
OPEN_ROOM[1:4, 1:4] = False


def make_level(**overrides) -> Level:
    kwargs = dict(
        walls=OPEN_ROOM,
        agent_start=(1, 1),
        objectives=(Objective((3, 3), value=1.0, feature_id=0),),
    )
    kwargs.update(overrides)
    return Level(**kwargs)  # type: ignore[arg-type]


def test_constructs_and_exposes_shape():
    level = make_level()
    assert level.shape == (5, 5)
    assert level.n_objectives == 1


def test_walls_are_copied_and_read_only():
    walls = OPEN_ROOM.copy()
    level = make_level(walls=walls)

    with pytest.raises(ValueError):
        level.walls[1, 1] = True

    # Mutating the caller's array must not affect the Level.
    walls[1, 1] = True
    assert not level.walls[1, 1]


def test_rejects_non_boolean_walls():
    with pytest.raises(TypeError, match="dtype bool"):
        make_level(walls=OPEN_ROOM.astype(np.uint8))


def test_rejects_non_2d_walls():
    with pytest.raises(ValueError, match="2-dimensional"):
        make_level(walls=OPEN_ROOM[None])


def test_rejects_agent_inside_a_wall():
    with pytest.raises(ValueError, match="agent_start.*inside a wall"):
        make_level(agent_start=(0, 0))


def test_rejects_agent_outside_the_maze():
    with pytest.raises(ValueError, match="outside"):
        make_level(agent_start=(9, 9))


def test_rejects_objective_inside_a_wall():
    with pytest.raises(ValueError, match=r"objectives\[0\].*inside a wall"):
        make_level(objectives=(Objective((0, 0), value=1.0, feature_id=0),))


def test_rejects_no_objectives():
    with pytest.raises(ValueError, match="at least one objective"):
        make_level(objectives=())


def test_rejects_objective_on_the_agent_start():
    with pytest.raises(ValueError, match="distinct"):
        make_level(
            agent_start=(1, 1),
            objectives=(Objective((1, 1), value=1.0, feature_id=0),),
        )


def test_rejects_coincident_objectives():
    with pytest.raises(ValueError, match="distinct"):
        make_level(
            objectives=(
                Objective((3, 3), value=1.0, feature_id=0),
                Objective((3, 3), value=0.5, feature_id=1),
            )
        )


def test_rejects_negative_feature_id():
    with pytest.raises(ValueError, match="feature_id"):
        Objective((1, 1), value=1.0, feature_id=-1)


def test_free_positions_lists_every_open_cell():
    level = make_level()
    assert set(level.free_positions()) == {(row, col) for row in range(1, 4) for col in range(1, 4)}


def test_is_wall():
    level = make_level()
    assert level.is_wall((0, 0))
    assert not level.is_wall((2, 2))


def test_objectives_are_normalised_to_a_tuple():
    level = make_level(objectives=[Objective((3, 3), value=1.0, feature_id=0)])
    assert isinstance(level.objectives, tuple)
