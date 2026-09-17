"""Stage 4: objectives behind crafting chains.

The stage-2 task hands the player every pickaxe and asks which ore to walk to.
Here the player starts bare-handed on a field with trees and stone deposits
(the walls are bedrock, see ``blocks.BEDROCK``), and an ore is worth its value
only after the chain that unlocks it: coal needs a wood pickaxe (three wood:
two for a table, one for the pickaxe); iron needs a stone pickaxe on top (one
more wood, one stone, and a return to the table). The trade-off
is still ``value - step_penalty x actions``, but the cost of an objective is
now the cost of a *plan*, most of which is shared between the two.

That is what makes this the stage that can separate a gain on the ore's
input cue from a value computed downstream: raising what iron is worth has
to raise the worth of the stone pickaxe, the stone and the fourth tree, none
of which carry the iron cue.

Three parts, all numpy: :class:`Field` and :class:`FieldSampler` (the world),
:class:`CraftTask` (tiles, observation, and the planner), and
:class:`CraftDemoSet` (the demonstration store, a self-contained
implementation of :class:`~goalmisgen.offline.demonstrations.Demonstrations`).
Routes are executed by :mod:`goalmisgen.craftax.engine`, unchanged.

**The planner is a canonical greedy policy.** Every decision it makes is a
function of the map, read the way a model emitting the route forwards can
read it: go to the *nearest* tree (ties by row-major cell order), cut it,
repeat; once three wood are in hand put the table on the tree just cut and
craft the wood pickaxe; for iron, take the nearer of the fourth tree and the
nearest stone, then the other, walk back beside the table, craft the stone
pickaxe; then the ore. Legs follow :mod:`goalmisgen.craftax.routes`: shortest
paths with one fixed move preference. The first version ranked tree orders
over permutations of static distances - a few actions cheaper, and not
learnable: the model could not tell which tree came first and hedged on the
first move (3% reach at 85% token accuracy). Cost is the greedy plan's
length; the choice between objectives is still ``value - step_penalty x cost``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Sequence

import numpy as np

from goalmisgen.craftax import simulate
from goalmisgen.craftax.blocks import BEDROCK, SOLID_BLOCKS, Action, Block
from goalmisgen.craftax.demos import MOVE_TO_ACTION, solution_from_costs
from goalmisgen.craftax.levels import OreFieldGenerator, _module_fingerprint
from goalmisgen.craftax.routes import canonical_moves, nearest
from goalmisgen.envs.dataset import _without_docstrings
from goalmisgen.envs.features import CorrelatedFeatures, FeatureScheme
from goalmisgen.envs.level import Objective, Position
from goalmisgen.envs.solver import MOVES, UNREACHABLE, LevelSolution, distance_field
from goalmisgen.envs.values import FixedValues, ValueScheme
from goalmisgen.offline.demonstrations import TASK_FILE, register_task
from goalmisgen.offline.demos import NO_ACTION
from goalmisgen.parallel import worker_pool

TASK = "craftax-craft"

WALL_CHANNEL, AGENT_CHANNEL, TREE_CHANNEL, STONE_CHANNEL, TABLE_CHANNEL, FIRST_KIND_CHANNEL = 0, 1, 2, 3, 4, 5
"""Observation layout: the maze's, with tree, stone-deposit and table channels before the kinds.

The table channel is what makes the state Markov once a table is down: without
it a placed table rendered as grass, the coal chain (which never returns to
the table) still worked, and the iron chain failed at exactly the walk back
to a table the policy could not see."""

STATE_PLANES = 4 + len(simulate.INVENTORY)
"""Receding-horizon observations append the player's facing (one-hot over the four
moves) and inventory (wood, stone, wood pickaxe, stone pickaxe) as planes constant
over the grid, after the value channel. Constant planes rather than a separate
token so the route model, the trainer, the decoders and the probes read the
observation unchanged; every cell's embedding sees the state."""

COUNT_SCALE = 4.0
"""Inventory counts are divided by this: the expert never holds more than four wood."""

_NEIGHBOURS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


@dataclasses.dataclass(frozen=True)
class Recipe:
    wood: int
    stone: int
    tool: str


RECIPES: dict[int, Recipe] = {
    int(Block.COAL): Recipe(wood=3, stone=0, tool="wood_pickaxe"),
    int(Block.IRON): Recipe(wood=4, stone=1, tool="stone_pickaxe"),
}
"""What each ore's chain consumes. Table: 2 wood. Wood pickaxe: 1 wood. Stone pickaxe: 1 wood + 1 stone."""

TABLE_AFTER = 3
"""Trees cut before the table goes down: two for the table, one for the wood pickaxe."""


# ----------------------------------------------------------------------
# The world
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Field:
    """One crafting episode: stone, trees, a bare-handed player, and the ores."""

    walls: np.ndarray
    """Bedrock: impassable and inert."""

    trees: np.ndarray
    stones: np.ndarray
    """Stone deposits, the only minable stone on the field."""

    agent_start: Position
    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        for name in ("walls", "trees", "stones"):
            grid = getattr(self, name)
            if grid.ndim != 2 or grid.dtype != np.bool_:
                raise TypeError(f"{name} must be a 2-d bool array")
            frozen = grid.copy()
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)
        if not (self.walls.shape == self.trees.shape == self.stones.shape) or self.walls.shape[0] != self.walls.shape[1]:
            raise ValueError(f"fields are square and walls/trees/stones agree in shape, got {self.walls.shape}")
        object.__setattr__(self, "objectives", tuple(self.objectives))
        if (self.walls & self.trees).any() or (self.walls & self.stones).any() or (self.trees & self.stones).any():
            raise ValueError("a cell holds at most one of bedrock, tree and stone")
        cells = [self.agent_start, *(o.position for o in self.objectives)]
        if len(set(cells)) != len(cells):
            raise ValueError("agent and objectives must occupy distinct cells")
        for cell in cells:
            if self.walls[cell] or self.trees[cell] or self.stones[cell]:
                raise ValueError(f"{cell} is not free")

    @property
    def shape(self) -> tuple[int, int]:
        return self.walls.shape  # type: ignore[return-value]

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    def solid(self) -> np.ndarray:
        """Cells the player cannot stand on: bedrock, trees, stone deposits, and the ores themselves."""
        solid = self.walls | self.trees | self.stones
        for objective in self.objectives:
            solid[objective.position] = True
        return solid


def _all_reachable(field: Field) -> bool:
    """Every tree and objective has a walkable neighbour the player can reach."""
    solid = field.solid()
    distances = distance_field(solid, field.agent_start)
    height, width = solid.shape
    targets = [tuple(int(v) for v in cell) for cell in np.argwhere(field.trees | field.stones)] + [
        o.position for o in field.objectives
    ]
    for r, c in targets:
        if not any(
            0 <= r + dr < height and 0 <= c + dc < width and distances[r + dr, c + dc] != UNREACHABLE for dr, dc in MOVES
        ):
            return False
    return True


@dataclasses.dataclass(frozen=True)
class FieldSampler:
    """Ore fields with trees, conditioned on every objective having a plan."""

    size: int = 15
    obstacle_density: float = 0.2
    n_trees: int = 6
    n_stones: int = 4
    """Stone deposits; the walls are bedrock and cannot be mined."""

    n_objectives: int = 2
    values: ValueScheme = dataclasses.field(default_factory=lambda: FixedValues((1.1, 0.5)))
    """A gap of 0.6 is 12 actions at the default step penalty: the median extra
    cost of the greedy iron chain over the greedy coal chain on these fields
    (p25/50/75 = 8/12.5/19), so the expert takes iron on about half and the
    +-0.45 arms (thresholds 3..21) sweep most of the distribution."""

    features: FeatureScheme = dataclasses.field(default_factory=CorrelatedFeatures)
    max_sampling_attempts: int = 200

    def __post_init__(self) -> None:
        if self.n_objectives != 2:
            raise ValueError("the crafting task is written for two objectives")

    def sample(self, rng: np.random.Generator) -> Field:
        """A field whose trees and ores can all be reached; whether both ores have a *plan* is checked at generation."""
        generator = OreFieldGenerator(self.obstacle_density)
        for _ in range(self.max_sampling_attempts):
            walls = generator.generate((self.size, self.size), rng)
            free = np.argwhere(~walls)
            needed = 1 + self.n_objectives + self.n_trees + self.n_stones
            if len(free) < needed:
                continue
            chosen = free[rng.choice(len(free), size=needed, replace=False)]
            agent = (int(chosen[0][0]), int(chosen[0][1]))
            objective_cells = [(int(r), int(c)) for r, c in chosen[1 : 1 + self.n_objectives]]
            trees = np.zeros_like(walls)
            for r, c in chosen[1 + self.n_objectives : 1 + self.n_objectives + self.n_trees]:
                trees[r, c] = True
            stones = np.zeros_like(walls)
            for r, c in chosen[1 + self.n_objectives + self.n_trees :]:
                stones[r, c] = True
            values = self.values.sample(self.n_objectives, rng)
            feature_ids = self.features.assign(values, rng)
            field = Field(
                walls=walls,
                trees=trees,
                stones=stones,
                agent_start=agent,
                objectives=tuple(
                    Objective(position=p, value=v, feature_id=f) for p, v, f in zip(objective_cells, values, feature_ids)
                ),
            )
            if _all_reachable(field):
                return field
        raise RuntimeError(f"no usable {self.size}x{self.size} field after {self.max_sampling_attempts} attempts")

    def canonical(self) -> "FieldSampler":
        return dataclasses.replace(self, features=self.features.canonical())


# ----------------------------------------------------------------------
# The task: tiles, observation, planner
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Plan:
    actions: tuple[int, ...]

    @property
    def cost(self) -> int:
        return len(self.actions)


@dataclasses.dataclass(frozen=True)
class CraftTask:
    receding: bool = True
    """Receding-horizon execution: the observation carries facing and inventory,
    training samples every state along a route, and decoding emits one action
    per forward pass with the engine in the loop. ``False`` is the open-loop
    prefix-LM of stage 2, kept for comparison."""

    kinds: tuple[int, ...] = (int(Block.IRON), int(Block.COAL))
    """Kind of feature 0 first. At rho=1 feature 0 marks the richer objective,
    so the expensive chain (iron) is the valuable one and the expert has a
    real trade-off to make; the cheap kind being the rich one would leave it
    nothing to decide."""

    start_direction: int = int(Action.DOWN)

    def __post_init__(self) -> None:
        for kind in self.kinds:
            if kind not in RECIPES:
                raise ValueError(f"no recipe for {Block(kind).name}")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("kinds must be distinct")

    @property
    def n_features(self) -> int:
        return len(self.kinds)

    @property
    def tools(self) -> tuple[str, ...]:
        return ()  # bare-handed: the chain is the task

    @property
    def interactable(self) -> tuple[int, ...]:
        return tuple(self.kinds) + (int(Block.TREE), int(Block.STONE))

    def tiles(self, field: Field) -> np.ndarray:
        tiles = np.where(field.walls, int(BEDROCK), int(Block.GRASS)).astype(np.int32)
        tiles[field.trees] = int(Block.TREE)
        tiles[field.stones] = int(Block.STONE)
        for objective in field.objectives:
            tiles[objective.position] = self.kinds[objective.feature_id]
        return tiles

    def n_channels(self, hide_values: bool = False) -> int:
        return FIRST_KIND_CHANNEL + self.n_features + (0 if hide_values else 1) + (STATE_PLANES if self.receding else 0)

    def observe(
        self,
        tiles: np.ndarray,
        agent: tuple[int, int],
        feature_values: Sequence[float],
        hide_values: bool = False,
        facing: int | None = None,
        inventory: Sequence[int] | None = None,
    ) -> np.ndarray:
        """The model's observation of one state. Facing and inventory default to the start of an episode."""
        facing = self.start_direction if facing is None else int(facing)
        inventory = (0,) * len(simulate.INVENTORY) if inventory is None else tuple(int(v) for v in inventory)
        return self.observe_batch(
            np.asarray(tiles)[None],
            np.asarray([agent]),
            np.asarray([feature_values]),
            hide_values,
            np.asarray([facing]),
            np.asarray([inventory]),
        )[0]

    def observe_batch(
        self,
        tiles: np.ndarray,
        agents: np.ndarray,
        feature_values: np.ndarray,
        hide_values: bool = False,
        facings: np.ndarray | None = None,
        inventories: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(B, H, W, C)`` observations for a batch of states, vectorised."""
        tiles = np.asarray(tiles)
        batch = tiles.shape[0]
        observation = np.zeros(tiles.shape + (self.n_channels(hide_values),), dtype=np.float32)
        observation[..., WALL_CHANNEL] = tiles == int(BEDROCK)
        rows = np.arange(batch)
        agents = np.asarray(agents)
        observation[rows, agents[:, 0], agents[:, 1], AGENT_CHANNEL] = 1.0
        observation[..., TREE_CHANNEL] = tiles == int(Block.TREE)
        observation[..., STONE_CHANNEL] = tiles == int(Block.STONE)
        observation[..., TABLE_CHANNEL] = tiles == int(Block.CRAFTING_TABLE)
        feature_values = np.asarray(feature_values, dtype=np.float32)
        for k, kind in enumerate(self.kinds):
            mask = tiles == kind
            plane = observation[..., FIRST_KIND_CHANNEL + k]
            plane[mask] = 1.0
            if not hide_values:
                value_plane = observation[..., FIRST_KIND_CHANNEL + self.n_features]
                value_plane[mask] = np.broadcast_to(feature_values[:, k, None, None], tiles.shape)[mask]
        if self.receding:
            first = FIRST_KIND_CHANNEL + self.n_features + (0 if hide_values else 1)
            facings = np.full(batch, self.start_direction) if facings is None else np.asarray(facings)
            inventories = np.zeros((batch, len(simulate.INVENTORY))) if inventories is None else np.asarray(inventories)
            for m, move in enumerate((Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)):
                observation[..., first + m] = (facings == int(move))[:, None, None]
            observation[..., first + 4 : first + 4 + len(simulate.INVENTORY)] = (inventories / COUNT_SCALE)[:, None, None, :]
        return observation

    # --- planning ------------------------------------------------------------

    def initial_state(self, field: Field) -> simulate.State:
        return simulate.initial(self.tiles(field), field.agent_start, self.start_direction, self.tools)

    def plans_from(self, state: simulate.State, field: Field) -> tuple[Plan | None, ...]:
        """The greedy plan for every objective from an arbitrary state of ``field`` (``None`` where there is none)."""
        planner = _Planner(self, state, field.objectives)
        return tuple(planner.best(index) for index in range(field.n_objectives))

    def plans(self, field: Field) -> tuple[Plan | None, ...]:
        """The greedy plan for every objective from the start of the episode."""
        return self.plans_from(self.initial_state(field), field)

    def plan(self, field: Field, index: int) -> Plan | None:
        return self.plans(field)[index]

    def solution_from(
        self, state: simulate.State, field: Field, step_penalty: float, step_limit: int | None = None
    ) -> LevelSolution:
        """The expert's verdict from an arbitrary state: value minus the cost of the plan from here."""
        return solution_from_costs(
            field, [None if p is None else p.cost for p in self.plans_from(state, field)], step_penalty, step_limit
        )

    def route_to(self, field: Field, index: int, width: int | None = None) -> np.ndarray | None:
        return route_array(self.plan(field, index), width)

    def costs(self, field: Field) -> tuple[int | None, ...]:
        return tuple(None if p is None else p.cost for p in self.plans(field))

    def solution(self, field: Field, step_penalty: float, step_limit: int | None = None) -> LevelSolution:
        return solution_from_costs(field, self.costs(field), step_penalty, step_limit)

    def to_json(self) -> dict:
        return {
            "kinds": list(self.kinds),
            "start_direction": self.start_direction,
            "planner": "greedy-canonical",
            "receding": self.receding,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CraftTask":
        if data.get("planner", "greedy-canonical") != "greedy-canonical":
            raise ValueError(f"demonstrations were planned by {data['planner']!r}, which this code no longer has")
        return cls(
            kinds=tuple(int(k) for k in data["kinds"]),
            start_direction=int(data["start_direction"]),
            receding=bool(data.get("receding", False)),
        )


def route_array(plan: Plan | None, width: int | None = None) -> np.ndarray | None:
    """A plan as a ``NO_ACTION``-padded action row, ``width`` wide."""
    if plan is None:
        return None
    width = plan.cost if width is None else width
    out = np.full(width, NO_ACTION, dtype=np.int32)
    out[: plan.cost] = plan.actions
    return out


_ACTION_TO_MOVE = {MOVE_TO_ACTION[i]: i for i in range(len(MOVES))}


class _Planner:
    """The greedy policy from any state, executed on the grid as it changes.

    Stated in terms of what the player still *needs* given the tiles, position,
    facing and inventory it has, so the start of an episode is the special case
    with nothing in hand and every mid-route state - or a counterfactual one,
    wood handed over for free - plans the same way. Along an expert route the
    plan from state ``t`` is exactly the remaining route (a test holds this).
    """

    def __init__(self, task: CraftTask, state: simulate.State, objectives: Sequence[Objective]) -> None:
        self.task = task
        self.state = state
        self.objectives = tuple(objectives)

    def best(self, index: int) -> Plan | None:
        objective = self.objectives[index]
        kind = self.task.kinds[objective.feature_id]
        recipe = RECIPES[kind]
        tiles = self.state.tiles
        if int(tiles[objective.position]) != kind:
            return None  # already mined, or not this objective's tile
        grid = np.isin(tiles, [int(b) for b in SOLID_BLOCKS])
        pos = self.state.position
        facing = _ACTION_TO_MOVE[self.state.facing]
        wood, stone, has_wp, has_sp = (int(v) for v in self.state.inventory)
        trees = [(int(r), int(c)) for r, c in np.argwhere(tiles == int(Block.TREE))]
        stones = [(int(r), int(c)) for r, c in np.argwhere(tiles == int(Block.STONE))]
        tables = [(int(r), int(c)) for r, c in np.argwhere(tiles == int(Block.CRAFTING_TABLE))]
        actions: list[int] = []

        def go(cell) -> bool:
            """Walk to ``cell`` (solid: end beside it, facing it) with the canonical route."""
            nonlocal pos, facing
            moves = canonical_moves(grid, pos, cell)
            if moves is None:
                return False
            end = pos
            for m in moves[:-1] if grid[cell] else moves:
                end = (end[0] + MOVES[m][0], end[1] + MOVES[m][1])
            if grid[cell] and moves and moves[-1] == (moves[-2] if len(moves) >= 2 else facing):
                moves = moves[:-1]  # already facing the cell after a straight approach
            actions.extend(MOVE_TO_ACTION[m] for m in moves)
            if moves:
                facing = moves[-1]
            pos = end
            return True

        def cut() -> tuple[int, int] | None:
            nonlocal wood
            found = nearest(grid, pos, trees)
            if found is None or not go(found[0]):
                return None
            actions.append(int(Action.DO))
            grid[found[0]] = False
            trees.remove(found[0])
            wood += 1
            return found[0]

        def mine_stone() -> bool:
            nonlocal stone
            found = nearest(grid, pos, stones)
            if found is None or not go(found[0]):
                return False
            actions.append(int(Action.DO))
            grid[found[0]] = False
            stones.remove(found[0])
            stone += 1
            return True

        def beside(table) -> bool:
            cells = [
                (table[0] + dr, table[1] + dc)
                for dr, dc in _NEIGHBOURS8
                if 0 <= table[0] + dr < grid.shape[0]
                and 0 <= table[1] + dc < grid.shape[1]
                and not grid[table[0] + dr, table[1] + dc]
            ]
            if pos in cells:
                return True
            found = nearest(grid, pos, cells)
            return found is not None and go(found[0])

        # What the chain still needs from this state.
        need_sp = recipe.stone > 0 and not has_sp
        need_wp = not has_wp and (recipe.stone == 0 or (need_sp and stone == 0))
        need_table = (need_wp or need_sp) and not tables
        table = None if not tables else nearest(grid, pos, tables)
        table = None if table is None else table[0]

        # Wood for the table and the wood pickaxe; then the table on the faced tile
        # if it is free (it is, right after a cut), else on a freshly cut tree's tile.
        if need_table:
            while wood < 2 + int(need_wp):
                if cut() is None:
                    return None
            faced = (pos[0] + MOVES[facing][0], pos[1] + MOVES[facing][1])
            usable = 0 <= faced[0] < grid.shape[0] and 0 <= faced[1] < grid.shape[1] and not grid[faced]
            if usable:
                # A table on a free tile can wall the player into a pocket; a tree's
                # tile never can, since the tree was solid already. Use the faced tile
                # only if everything the plan still needs stays reachable.
                grid[faced] = True
                needed = (
                    [objective.position]
                    + (trees if (wood - 2 < int(need_wp) + int(need_sp)) else [])
                    + (stones if need_sp and stone < 1 else [])
                )
                usable = all(nearest(grid, pos, [cell]) is not None for cell in needed)
                grid[faced] = False
            if not usable:
                faced = cut()
                if faced is None:
                    return None
            actions.append(int(Action.PLACE_TABLE))
            grid[faced] = True
            table = faced
            wood -= 2
        if need_wp:
            while wood < 1:
                if cut() is None:
                    return None
            if not beside(table):
                return None
            actions.append(int(Action.MAKE_WOOD_PICKAXE))
            wood -= 1
        if need_sp:
            # The nearer of what is still missing first, a tie to the tree.
            while wood < 1 or stone < 1:
                tree = nearest(grid, pos, trees) if wood < 1 else None
                deposit = nearest(grid, pos, stones) if stone < 1 else None
                if tree is None and deposit is None:
                    return None
                if deposit is None or (tree is not None and tree[1] <= deposit[1]):
                    if cut() is None:
                        return None
                elif not mine_stone():
                    return None
            if not beside(table):
                return None
            actions.append(int(Action.MAKE_STONE_PICKAXE))
            wood -= 1
            stone -= 1
        if not go(objective.position):
            return None
        actions.append(int(Action.DO))
        return Plan(tuple(actions))


# ----------------------------------------------------------------------
# The demonstration store
# ----------------------------------------------------------------------

ARRAY_FIELDS: tuple[str, ...] = (
    "level_index",
    "walls_packed",
    "trees_packed",
    "stones_packed",
    "agent",
    "positions",
    "values",
    "feature_ids",
    "actions",
    "lengths",
    "distances",
    "target",
    "ambiguous",
    "utility_margin",
)


def pool_fingerprint(sampler: FieldSampler, seed: int) -> str:
    """Identifies a pool of fields: sampler config, seed, and the code that draws and plans them."""
    digest = hashlib.sha256()
    digest.update(repr(sampler.canonical()).encode())
    digest.update(str(seed).encode())
    digest.update(_module_fingerprint().encode())
    source = pathlib.Path(__file__).read_text()
    import ast

    digest.update(hashlib.sha256(ast.dump(_without_docstrings(ast.parse(source))).encode()).hexdigest().encode())
    return digest.hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class CraftDemoSet:
    """Expert crafting demonstrations on a pool of fields, at one correlation."""

    level_index: np.ndarray
    walls_packed: np.ndarray
    trees_packed: np.ndarray
    stones_packed: np.ndarray
    agent: np.ndarray
    positions: np.ndarray
    values: np.ndarray
    feature_ids: np.ndarray
    actions: np.ndarray
    lengths: np.ndarray
    distances: np.ndarray
    target: np.ndarray
    ambiguous: np.ndarray
    utility_margin: np.ndarray
    size: int
    meta: dict
    task: CraftTask = dataclasses.field(default_factory=CraftTask)
    path: pathlib.Path | None = None
    hide_values: bool = False

    def __len__(self) -> int:
        return len(self.level_index)

    @property
    def n_objectives(self) -> int:
        return self.positions.shape[1]

    @property
    def n_channels(self) -> int:
        return self.task.n_channels(self.hide_values)

    @property
    def n_actions(self) -> int:
        return len(Action)

    @property
    def max_actions(self) -> int:
        return self.actions.shape[1]

    @property
    def move_actions(self) -> tuple[int, ...]:
        return MOVE_TO_ACTION

    @property
    def rho(self) -> float:
        return float(self.meta["rho"])

    def _grids(self, name: str, indices) -> np.ndarray:
        packed = np.asarray(getattr(self, name)[np.asarray(indices)])
        flat = np.unpackbits(packed, axis=1, count=self.size * self.size)
        return flat.reshape(len(packed), self.size, self.size).astype(np.bool_)

    def level(self, index: int) -> Field:
        return Field(
            walls=self._grids("walls_packed", [index])[0],
            trees=self._grids("trees_packed", [index])[0],
            stones=self._grids("stones_packed", [index])[0],
            agent_start=(int(self.agent[index, 0]), int(self.agent[index, 1])),
            objectives=tuple(
                Objective(
                    position=(int(self.positions[index, k, 0]), int(self.positions[index, k, 1])),
                    value=float(self.values[index, k]),
                    feature_id=int(self.feature_ids[index, k]),
                )
                for k in range(self.n_objectives)
            ),
        )

    def feature_values(self, indices) -> np.ndarray:
        """``(B, K)`` value of each feature id, the order ``observe`` wants."""
        indices = np.asarray(indices)
        out = np.zeros((len(indices), self.task.n_features), dtype=np.float32)
        rows = np.arange(len(indices))
        for k in range(self.n_objectives):
            out[rows, self.feature_ids[indices, k]] = self.values[indices, k]
        return out

    def tiles(self, indices) -> np.ndarray:
        """``(B, size, size)`` int32 block ids of the fields at the start of the episode."""
        indices = np.asarray(indices)
        tiles = np.where(self._grids("walls_packed", indices), int(BEDROCK), int(Block.GRASS)).astype(np.int32)
        tiles[self._grids("trees_packed", indices)] = int(Block.TREE)
        tiles[self._grids("stones_packed", indices)] = int(Block.STONE)
        rows = np.arange(len(indices))
        for k in range(self.n_objectives):
            r, c = self.positions[indices, k, 0], self.positions[indices, k, 1]
            tiles[rows, r, c] = np.asarray(self.task.kinds)[self.feature_ids[indices, k]]
        return tiles

    def observations(self, indices) -> np.ndarray:
        """Observations before the first action, ``(B, size, size, C)``."""
        indices = np.asarray(indices)
        return self.task.observe_batch(
            self.tiles(indices), np.asarray(self.agent[indices]), self.feature_values(indices), self.hide_values
        )

    def decode_closed_loop(self, model, params, indices):
        """Receding-horizon decoding through the engine; what ``decode.evaluate`` uses when the task is receding.

        ``None`` (the attribute, not the call) on an open-loop task, so the
        dispatch in ``evaluate`` falls through to ``greedy_decode``.
        """
        from goalmisgen.craftax.closed_loop import rollout

        return rollout(model, params, self, np.asarray(indices))

    def suffixes(self, after_pickaxe_weight: int = 1) -> "SuffixDemoSet":
        """Every state along every route as a training item; see :class:`SuffixDemoSet`."""
        return SuffixDemoSet.of(self, after_pickaxe_weight)

    def routes(self, indices) -> np.ndarray:
        return np.asarray(self.actions[np.asarray(indices)]).astype(np.int32)

    def replay(self, index: int, actions: Sequence[int], emitted_eos: bool = True) -> dict:
        from goalmisgen.craftax import engine

        return engine.replay(
            self.level(index),
            self.task,
            np.asarray(actions),
            float(self.meta["step_penalty"]),
            int(self.meta["step_limit"]),
            emitted_eos,
        )

    def replay_many(self, indices, actions, emitted_eos) -> list[dict]:
        from goalmisgen.craftax import engine

        return engine.replay_batch(
            [self.level(int(i)) for i in indices],
            self.task,
            np.asarray(actions),
            float(self.meta["step_penalty"]),
            int(self.meta["step_limit"]),
            list(emitted_eos),
        )

    # --- views ---
    def subset(self, indices) -> "CraftDemoSet":
        indices = np.asarray(indices)
        arrays = {name: np.asarray(getattr(self, name)[indices]) for name in ARRAY_FIELDS}
        return dataclasses.replace(self, **arrays, meta={**self.meta, "n": int(len(indices))}, path=None)

    def with_hidden_values(self, hide: bool = True) -> "CraftDemoSet":
        return dataclasses.replace(self, hide_values=hide)

    def with_values(self, values) -> "CraftDemoSet":
        values = np.asarray(values)
        if values.shape != self.values.shape:
            raise ValueError(f"values must be shaped {self.values.shape}, got {values.shape}")
        return dataclasses.replace(self, values=values.copy())

    def with_feature_ids(self, feature_ids) -> "CraftDemoSet":
        feature_ids = np.asarray(feature_ids)
        if feature_ids.shape != self.feature_ids.shape:
            raise ValueError(f"feature_ids must be shaped {self.feature_ids.shape}, got {feature_ids.shape}")
        return dataclasses.replace(self, feature_ids=feature_ids.copy())

    # --- persistence ---
    def save(self, path) -> None:
        directory = pathlib.Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ARRAY_FIELDS:
            np.save(directory / f"{name}.npy", np.asarray(getattr(self, name)))
        (directory / "meta.json").write_text(json.dumps({**self.meta, "size": self.size, "n": len(self)}, indent=2))
        (directory / TASK_FILE).write_text(json.dumps({"task": TASK, **self.task.to_json()}, indent=2))

    def __post_init__(self) -> None:
        if not self.task.receding:
            object.__setattr__(self, "decode_closed_loop", None)

    @classmethod
    def load(cls, path, mmap: bool = True, hide_values: bool = False) -> "CraftDemoSet":
        directory = pathlib.Path(path)
        meta = json.loads((directory / "meta.json").read_text())
        marker = json.loads((directory / TASK_FILE).read_text())
        if marker.get("task") != TASK:
            raise ValueError(f"{directory} holds task {marker.get('task')!r}, not {TASK!r}")
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None) for name in ARRAY_FIELDS}
        return cls(
            **arrays,
            size=int(meta["size"]),
            meta=meta,
            task=CraftTask.from_json(marker),
            path=directory,
            hide_values=hide_values,
        )

    # --- generation ---
    @classmethod
    def generate(
        cls,
        sampler: FieldSampler,
        seed: int,
        start: int,
        count: int,
        rho: float,
        task: CraftTask | None = None,
        colour_seed: int = 0,
        step_penalty: float = 0.05,
        step_limit: int = 200,
        max_actions: int = 128,
        workers: int = 1,
        chunk_size: int = 2_000,
        split: str | None = None,
    ) -> "CraftDemoSet":
        """Fields ``start .. start+count`` of the pool ``(sampler, seed)``, demonstrated at ``rho``.

        Field ``i`` is drawn from the ``i``-th child of ``SeedSequence(seed)``,
        so a pool is addressed by index the way a level dataset is: splits are
        index ranges, and ``shared_levels`` can compare sets from one pool.
        Kinds are assigned from ``(colour_seed, i)``, so sets at different rho
        are paired field for field; fixed values consume no randomness, so
        pools at different values share layouts, as the maze's do.
        """
        task = task or CraftTask()
        ranges = [(s, min(s + chunk_size, start + count)) for s in range(start, start + count, chunk_size)]
        args = [(sampler, seed, lo, hi, rho, task, colour_seed, step_penalty, step_limit, max_actions) for lo, hi in ranges]
        if workers > 1 and len(args) > 1:
            with worker_pool(workers) as pool:
                blocks = pool.starmap(demonstrate_block, args)
        else:
            blocks = [demonstrate_block(*a) for a in args]
        arrays = {name: np.concatenate([b[name] for b in blocks]) for name in ARRAY_FIELDS}
        values = sorted({float(v) for v in np.unique(arrays["values"])}, reverse=True)
        meta = {
            "task": {"task": TASK, **task.to_json()},
            "source": None,
            "source_fingerprint": pool_fingerprint(sampler, seed),
            "sampler": repr(sampler),
            "seed": int(seed),
            "start": int(start),
            "split": split,
            "rho": float(rho),
            "colour_seed": int(colour_seed),
            "values": values,
            "step_penalty": float(step_penalty),
            "step_limit": int(step_limit),
            "max_actions": int(max_actions),
            "n": int(count),
        }
        return cls(**arrays, size=sampler.size, meta=meta, task=task)


def demonstrate_block(sampler, seed, lo, hi, rho, task, colour_seed, step_penalty, step_limit, max_actions) -> dict:
    """Fields ``lo .. hi`` of the pool, planned. Safe to run in a worker process."""
    children = np.random.SeedSequence(seed).spawn(hi)[lo:hi]
    scheme = CorrelatedFeatures(rho)
    canonical = sampler.canonical()
    count = hi - lo
    n_objectives = sampler.n_objectives
    size = sampler.size
    out = {
        "level_index": np.arange(lo, hi, dtype=np.int64),
        "walls_packed": np.empty((count, int(np.ceil(size * size / 8))), dtype=np.uint8),
        "trees_packed": np.empty((count, int(np.ceil(size * size / 8))), dtype=np.uint8),
        "stones_packed": np.empty((count, int(np.ceil(size * size / 8))), dtype=np.uint8),
        "agent": np.empty((count, 2), dtype=np.uint8),
        "positions": np.empty((count, n_objectives, 2), dtype=np.uint8),
        "values": np.empty((count, n_objectives), dtype=np.float64),
        "feature_ids": np.empty((count, n_objectives), dtype=np.int8),
        "actions": np.full((count, max_actions), NO_ACTION, dtype=np.int8),
        "lengths": np.empty(count, dtype=np.int16),
        "distances": np.full((count, n_objectives), -1, dtype=np.int16),
        "target": np.empty(count, dtype=np.int8),
        "ambiguous": np.empty(count, dtype=np.bool_),
        "utility_margin": np.empty(count, dtype=np.float32),
    }
    for row, (index, child) in enumerate(zip(range(lo, hi), children)):
        rng = np.random.default_rng(child)
        for _ in range(sampler.max_sampling_attempts):
            layout = canonical.sample(rng)  # canonical features: the layout must not depend on rho
            values = tuple(o.value for o in layout.objectives)
            feature_ids = scheme.assign(values, np.random.default_rng([int(colour_seed), int(index)]))
            field = Field(
                walls=layout.walls,
                trees=layout.trees,
                stones=layout.stones,
                agent_start=layout.agent_start,
                objectives=tuple(Objective(o.position, o.value, int(f)) for o, f in zip(layout.objectives, feature_ids)),
            )
            # Conditioned on both chains being plannable. The sampler already
            # guarantees every tree and ore is reachable, so rejection here is
            # rare (a tree boxed in by the table, say) - rare enough that pools
            # at different rho are paired on all but a handful of fields.
            plans = task.plans(field)
            if all(plan is not None for plan in plans):
                break
        else:
            raise RuntimeError(f"field {index}: no plannable layout after {sampler.max_sampling_attempts} attempts")
        solution = solution_from_costs(field, [p.cost for p in plans], step_penalty, step_limit)
        route = route_array(plans[solution.optimal_index])
        assert route is not None
        length = int((route >= 0).sum())
        if length > max_actions:
            raise ValueError(f"field {index} needs {length} actions but max_actions={max_actions}")
        out["walls_packed"][row] = np.packbits(field.walls.reshape(-1))
        out["trees_packed"][row] = np.packbits(field.trees.reshape(-1))
        out["stones_packed"][row] = np.packbits(field.stones.reshape(-1))
        out["agent"][row] = field.agent_start
        for k, objective in enumerate(field.objectives):
            out["positions"][row, k] = objective.position
            out["values"][row, k] = objective.value
            out["feature_ids"][row, k] = objective.feature_id
        out["actions"][row, :length] = route[:length]
        out["lengths"][row] = length
        out["distances"][row] = [-1 if c is None else c for c in solution.distances]
        out["target"][row] = solution.optimal_index
        out["ambiguous"][row] = solution.is_ambiguous
        out["utility_margin"][row] = solution.utility_margin if np.isfinite(solution.utility_margin) else np.inf
    return out


@dataclasses.dataclass(frozen=True)
class SuffixDemoSet:
    """A crafting set seen as (state, remaining route) pairs: receding-horizon training data.

    Item ``j`` is field ``field_of[j]`` at step ``t_of[j]`` of its expert
    route: the observation is that state (reconstructed by
    :mod:`goalmisgen.craftax.simulate`) and the route is what the expert did
    from there. Every suffix is a demonstration of the same policy, so the
    loss is unchanged; only the input distribution widens to the states a
    closed-loop policy will actually meet. Implements the parts of
    :class:`~goalmisgen.offline.demonstrations.Demonstrations` the trainer
    reads; per-item ground truth is the field's.
    """

    base: CraftDemoSet
    field_of: np.ndarray  # (M,) int64
    t_of: np.ndarray  # (M,) int32

    @classmethod
    def of(cls, base: CraftDemoSet, after_pickaxe_weight: int = 1) -> "SuffixDemoSet":
        """Every state along every route; states after the wood pickaxe repeated ``after_pickaxe_weight`` times.

        Routes are mostly walking, so sampled uniformly the crafting decisions
        - which all lie after the wood pickaxe - are a few tokens in a hundred
        of the loss, and they are where the trained policy fails. Repeating
        those states shifts the signal to where it is needed without changing
        what any state is labelled with.
        """
        lengths = np.asarray(base.lengths).astype(np.int64)
        field_of = np.repeat(np.arange(len(base), dtype=np.int64), lengths)
        t_of = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths]) if len(lengths) else np.zeros(0, np.int32)
        if after_pickaxe_weight > 1:
            routes = np.asarray(base.actions)
            crafted = routes == int(Action.MAKE_WOOD_PICKAXE)
            pickaxe_at = np.where(crafted.any(axis=1), crafted.argmax(axis=1), np.iinfo(np.int32).max)
            late = t_of > pickaxe_at[field_of]
            field_of = np.concatenate([field_of] + [field_of[late]] * (after_pickaxe_weight - 1))
            t_of = np.concatenate([t_of] + [t_of[late]] * (after_pickaxe_weight - 1))
        return cls(base=base, field_of=field_of, t_of=t_of)

    def __len__(self) -> int:
        return len(self.field_of)

    @property
    def size(self) -> int:
        return self.base.size

    @property
    def hide_values(self) -> bool:
        return self.base.hide_values

    @property
    def path(self):
        return self.base.path

    @property
    def meta(self) -> dict:
        return {**self.base.meta, "suffixes": True, "n_items": len(self)}

    @property
    def n_channels(self) -> int:
        return self.base.n_channels

    @property
    def n_actions(self) -> int:
        return self.base.n_actions

    @property
    def max_actions(self) -> int:
        return self.base.max_actions

    @property
    def move_actions(self):
        return self.base.move_actions

    @property
    def rho(self) -> float:
        return self.base.rho

    @property
    def task(self) -> CraftTask:
        return self.base.task

    @property
    def level_index(self) -> np.ndarray:
        return np.asarray(self.base.level_index)[self.field_of]

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.base.lengths)[self.field_of] - self.t_of

    def _field_array(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self.base, name))[self.field_of]

    values = property(lambda self: self._field_array("values"))
    distances = property(lambda self: self._field_array("distances"))
    feature_ids = property(lambda self: self._field_array("feature_ids"))
    target = property(lambda self: self._field_array("target"))
    ambiguous = property(lambda self: self._field_array("ambiguous"))
    utility_margin = property(lambda self: self._field_array("utility_margin"))
    agent = property(lambda self: self._field_array("agent"))
    positions = property(lambda self: self._field_array("positions"))

    def states(self, indices) -> list:
        """The simulated :class:`simulate.State` of each item."""
        indices = np.asarray(indices)
        fields = self.field_of[indices]
        routes = self.base.routes(fields)
        tiles = self.base.tiles(fields)
        out = []
        for row, (i, t) in enumerate(zip(fields, self.t_of[indices])):
            state = simulate.initial(
                tiles[row], tuple(int(v) for v in self.base.agent[i]), self.task.start_direction, self.task.tools
            )
            for action in routes[row, :t]:
                state = simulate.step(state, int(action))
            out.append(state)
        return out

    def observations(self, indices) -> np.ndarray:
        indices = np.asarray(indices)
        states = self.states(indices)
        return self.task.observe_batch(
            np.stack([s.tiles for s in states]),
            np.asarray([s.position for s in states]),
            self.base.feature_values(self.field_of[indices]),
            self.hide_values,
            np.asarray([s.facing for s in states]),
            np.asarray([s.inventory for s in states]),
        )

    def routes(self, indices) -> np.ndarray:
        indices = np.asarray(indices)
        full = self.base.routes(self.field_of[indices])
        out = np.full_like(full, NO_ACTION)
        for row, t in enumerate(self.t_of[indices]):
            n = full.shape[1] - t
            out[row, :n] = full[row, t:]
        return out

    def level(self, index: int):
        return self.base.level(int(self.field_of[index]))

    def replay(self, index: int, actions, emitted_eos: bool = True) -> dict:
        raise NotImplementedError("suffix items are training data; evaluate on the underlying CraftDemoSet")

    def subset(self, indices) -> "SuffixDemoSet":
        indices = np.asarray(indices)
        return dataclasses.replace(self, field_of=self.field_of[indices], t_of=self.t_of[indices])

    def with_hidden_values(self, hide: bool = True) -> "SuffixDemoSet":
        return dataclasses.replace(self, base=self.base.with_hidden_values(hide))

    def with_values(self, values):
        raise NotImplementedError("counterfactuals are built on the underlying CraftDemoSet")

    def with_feature_ids(self, feature_ids):
        raise NotImplementedError("counterfactuals are built on the underlying CraftDemoSet")

    def save(self, path) -> None:
        raise NotImplementedError("a suffix view is derived; save the underlying CraftDemoSet")


@register_task(TASK)
def _load(path: pathlib.Path, mmap: bool, hide_values: bool) -> CraftDemoSet:
    return CraftDemoSet.load(path, mmap=mmap, hide_values=hide_values)
