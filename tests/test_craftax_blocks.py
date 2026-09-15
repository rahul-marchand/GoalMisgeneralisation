"""The JAX-free mirror of Craftax-Classic's vocabulary matches the real engine.

``goalmisgen.craftax.blocks`` restates block ids, action ids, move vectors and
the solid set by value so the world builder and planner never import Craftax.
Craftax is pinned exactly in ``pyproject.toml``; this is what makes the pin
enforceable. The per-ore pickaxe rule is checked behaviourally in
``tests/test_craftax_engine.py``, against ``do_action`` itself.
"""

from __future__ import annotations

import numpy as np
from craftax.craftax_classic import constants as engine

from goalmisgen.craftax import blocks


def test_block_ids_match_the_engine():
    assert [(b.name, b.value) for b in blocks.Block] == [(b.name, b.value) for b in engine.BlockType]


def test_action_ids_match_the_engine():
    assert [(a.name, a.value) for a in blocks.Action] == [(a.name, a.value) for a in engine.Action]
    assert blocks.N_ACTIONS == len(engine.Action)


def test_move_vectors_match_the_engine():
    table = np.asarray(engine.DIRECTIONS)
    for action, (d_row, d_col) in blocks.DIRECTIONS.items():
        assert tuple(table[action.value]) == (d_row, d_col), action
    assert set(blocks.DIRECTIONS) == set(blocks.MOVE_ACTIONS)
    # The engine's table is shorter than the action set; JAX clamps the index,
    # so every row past the moves is a zero vector and the rest read as no move.
    non_moves = [a for a in blocks.Action if a not in blocks.DIRECTIONS and a.value < len(table)]
    assert all(tuple(table[a.value]) == (0, 0) for a in non_moves)


def test_solid_blocks_match_the_engine():
    assert {b.value for b in blocks.SOLID_BLOCKS} == set(np.asarray(engine.SOLID_BLOCKS).tolist())


def test_out_of_bounds_is_not_solid():
    # The engine lets the player walk onto OUT_OF_BOUNDS, so a world may never
    # rely on it as a border; it must pad with a solid block.
    assert blocks.Block.OUT_OF_BOUNDS not in blocks.SOLID_BLOCKS
    assert blocks.Block.OUT_OF_BOUNDS in blocks.IMPASSABLE_BLOCKS


def test_collectables_have_a_tool_a_product_and_an_inventory_field():
    for block in blocks.COLLECTABLE_BLOCKS:
        assert block in blocks.REQUIRED_TOOL
        assert block in blocks.MINED_TO
        assert block in blocks.INVENTORY_FIELD
        assert blocks.MINED_TO[block] not in blocks.SOLID_BLOCKS
    assert set(blocks.REQUIRED_TOOL.values()) - {None} <= set(blocks.TOOLS)
