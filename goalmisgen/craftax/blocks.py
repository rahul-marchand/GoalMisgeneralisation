"""Craftax-Classic's block and action vocabularies, mirrored without importing it.

``craftax.craftax_classic.constants`` imports JAX at module scope, and the
world builder and planner must stay JAX-free (see the package docstring). So
the handful of engine facts they depend on are restated here by value, and
``tests/test_craftax_blocks.py`` asserts every one of them against the real
enums. Craftax is pinned exactly in ``pyproject.toml`` for the same reason.
"""

from __future__ import annotations

import enum


class Block(enum.IntEnum):
    """``craftax.craftax_classic.constants.BlockType``, by value."""

    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16


class Action(enum.IntEnum):
    """``craftax.craftax_classic.constants.Action``, by value."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    DO = 5
    SLEEP = 6
    PLACE_STONE = 7
    PLACE_TABLE = 8
    PLACE_FURNACE = 9
    PLACE_PLANT = 10
    MAKE_WOOD_PICKAXE = 11
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16


N_ACTIONS = len(Action)

MOVE_ACTIONS: tuple[Action, ...] = (Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN)

DIRECTIONS: dict[Action, tuple[int, int]] = {
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
}
"""``(d_row, d_col)`` for each move, matching the engine's ``DIRECTIONS`` table.

A move into a solid block does not move the player but still turns them to face
it, which is how a route ends up facing the ore it is about to mine.
"""

SOLID_BLOCKS: frozenset[Block] = frozenset(
    {
        Block.WATER,
        Block.STONE,
        Block.TREE,
        Block.COAL,
        Block.IRON,
        Block.DIAMOND,
        Block.CRAFTING_TABLE,
        Block.FURNACE,
        Block.PLANT,
        Block.RIPE_PLANT,
    }
)
"""Blocks the player cannot walk through - the engine's ``SOLID_BLOCKS``."""

IMPASSABLE_BLOCKS: frozenset[Block] = SOLID_BLOCKS | {Block.LAVA, Block.INVALID, Block.OUT_OF_BOUNDS}
"""Blocks a route may not step on: solid ones, and lava, which ends the episode."""

REQUIRED_TOOL: dict[Block, str | None] = {
    Block.TREE: None,
    Block.STONE: "wood_pickaxe",
    Block.COAL: "wood_pickaxe",
    Block.IRON: "stone_pickaxe",
    Block.DIAMOND: "iron_pickaxe",
}
"""The inventory flag ``do_action`` checks before mining each collectable block.

Each ore checks only its own flag: an iron pickaxe alone does not mine coal.
"""

COLLECTABLE_BLOCKS: tuple[Block, ...] = tuple(REQUIRED_TOOL)

TOOLS: tuple[str, ...] = ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe")
"""Inventory fields a world may hand the player at the start."""

MINED_TO: dict[Block, Block] = {
    Block.TREE: Block.GRASS,
    Block.STONE: Block.PATH,
    Block.COAL: Block.PATH,
    Block.IRON: Block.PATH,
    Block.DIAMOND: Block.PATH,
}
"""What a collectable block becomes once mined."""

INVENTORY_FIELD: dict[Block, str] = {
    Block.TREE: "wood",
    Block.STONE: "stone",
    Block.COAL: "coal",
    Block.IRON: "iron",
    Block.DIAMOND: "diamond",
}
"""The inventory count that goes up by one when the block is mined."""
