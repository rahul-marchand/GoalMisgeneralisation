"""A numpy twin of the engine for the actions our experts use.

Receding-horizon training needs the state at every step of an expert route -
the map as it is by then, the player's position and facing, the inventory -
and storing those is a thousand times the size of the route. So they are
reconstructed on the fly by replaying the route with these rules, which are
the engine's for the eleven actions our experts emit (moves, DO on trees,
stone and ores, placing a table, crafting the two pickaxes). Every other
action is a no-op here; a decoded model can emit them, but decoded routes are
executed by the real engine, never by this. ``tests/test_craftax_simulate.py``
holds this equal to the engine, field by field, along expert routes.

Pure Python over small tuples on purpose: a training batch replays a few
hundred routes, and numpy per step would be slower than integer arithmetic.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

import numpy as np

from goalmisgen.craftax.blocks import DIRECTIONS, REQUIRED_TOOL, SOLID_BLOCKS, Action, Block

INVENTORY: tuple[str, ...] = ("wood", "stone", "wood_pickaxe", "stone_pickaxe")
"""The inventory fields the tasks can change, in the order the observation shows them."""

_SOLID = frozenset(int(b) for b in SOLID_BLOCKS)
_TABLE_NEIGHBOURHOOD = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
_CAP = 9  # the engine caps every inventory count


@dataclasses.dataclass(frozen=True)
class State:
    tiles: np.ndarray  # (H, W) int32, a private copy
    position: tuple[int, int]
    facing: int  # a move action id
    inventory: tuple[int, int, int, int]  # in INVENTORY order
    collected: int | None = None
    """The ore block id mined by the last action, if any: the episode ends there."""

    @property
    def wood(self) -> int:
        return self.inventory[0]

    @property
    def stone(self) -> int:
        return self.inventory[1]

    def faced(self) -> tuple[int, int]:
        d_row, d_col = DIRECTIONS[Action(self.facing)]
        return (self.position[0] + d_row, self.position[1] + d_col)


def initial(tiles: np.ndarray, agent: tuple[int, int], facing: int, tools: Iterable[str] = ()) -> State:
    inventory = [0, 0, 0, 0]
    for tool in tools:
        inventory[INVENTORY.index(tool)] = 1
    return State(np.array(tiles, dtype=np.int32, copy=True), (int(agent[0]), int(agent[1])), int(facing), tuple(inventory))


def step(state: State, action: int) -> State:
    """The engine's effect of ``action`` on ``state``, for the actions our experts use."""
    tiles = state.tiles
    height, width = tiles.shape
    action = int(action)
    inventory = list(state.inventory)
    position, facing = state.position, state.facing
    collected = None
    new_tiles = tiles

    def in_bounds(cell) -> bool:
        return 0 <= cell[0] < height and 0 <= cell[1] < width

    if action in (int(Action.LEFT), int(Action.RIGHT), int(Action.UP), int(Action.DOWN)):
        facing = action
        d_row, d_col = DIRECTIONS[Action(action)]
        nxt = (position[0] + d_row, position[1] + d_col)
        if in_bounds(nxt) and int(tiles[nxt]) not in _SOLID:
            position = nxt
    elif action == int(Action.DO):
        cell = state.faced()
        if in_bounds(cell):
            block = int(tiles[cell])
            if block == int(Block.TREE):
                new_tiles = tiles.copy()
                new_tiles[cell] = int(Block.GRASS)
                inventory[0] = min(_CAP, inventory[0] + 1)
            elif block in (int(Block.STONE), int(Block.COAL), int(Block.IRON)):
                tool = REQUIRED_TOOL[Block(block)]
                if inventory[INVENTORY.index(tool)] > 0:
                    new_tiles = tiles.copy()
                    new_tiles[cell] = int(Block.PATH)
                    if block == int(Block.STONE):
                        inventory[1] = min(_CAP, inventory[1] + 1)
                    else:
                        collected = block
    elif action == int(Action.PLACE_TABLE):
        cell = state.faced()
        if inventory[0] >= 2 and in_bounds(cell) and int(tiles[cell]) not in _SOLID:
            new_tiles = tiles.copy()
            new_tiles[cell] = int(Block.CRAFTING_TABLE)
            inventory[0] -= 2
    elif action in (int(Action.MAKE_WOOD_PICKAXE), int(Action.MAKE_STONE_PICKAXE)):
        near_table = any(
            in_bounds((position[0] + dr, position[1] + dc))
            and int(tiles[position[0] + dr, position[1] + dc]) == int(Block.CRAFTING_TABLE)
            for dr, dc in _TABLE_NEIGHBOURHOOD
        )
        if action == int(Action.MAKE_WOOD_PICKAXE) and near_table and inventory[0] >= 1:
            inventory[0] -= 1
            inventory[2] = min(_CAP, inventory[2] + 1)
        elif action == int(Action.MAKE_STONE_PICKAXE) and near_table and inventory[0] >= 1 and inventory[1] >= 1:
            inventory[0] -= 1
            inventory[1] -= 1
            inventory[3] = min(_CAP, inventory[3] + 1)
    return State(new_tiles, position, facing, tuple(inventory), collected)


def trajectory(start: State, route: Sequence[int]) -> list[State]:
    """States before each action of ``route`` (``NO_ACTION`` ends it), then the final state: ``len(route) + 1``."""
    states = [start]
    for action in route:
        if int(action) < 0:
            break
        states.append(step(states[-1], int(action)))
    return states
