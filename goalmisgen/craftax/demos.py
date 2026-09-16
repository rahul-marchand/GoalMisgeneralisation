"""The maze's demonstrations, executed as Craftax routes.

A stage-2 Craftax world *is* a maze level rendered into the engine (see
:mod:`goalmisgen.craftax.levels`), and the expert route is the maze route plus
one DO: the maze path's last move steps onto the objective, but in the engine
the ore is solid, so that move only turns the player to face it, and DO then
mines it. When the approach is straight the player already faces the ore and
that turn would be a wasted action, so it is dropped: the engine route to an
objective costs ``d_maze + 1`` actions after a corner and ``d_maze`` after a
straight run. The expert is optimal *in the engine*: every objective is costed
by its own engine route and chosen by ``value - step_penalty x cost``, which
can differ from the maze's choice only on near-ties.

So a Craftax demonstration set is a :class:`~goalmisgen.offline.demos.DemoSet`
seen through a :class:`CraftaxTask`: the same levels and observation, plus the
engine routes, costs and choices stored beside them. A set at rho here is
paired level for level with the maze's at the same rho.

The one thing this module cannot do is execute a route: that is the engine's
job, in :mod:`goalmisgen.craftax.engine`, imported only inside :meth:`replay`
so that generation - which runs in a spawned worker pool - never loads JAX.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Sequence

import numpy as np

from goalmisgen.craftax.blocks import (
    BEDROCK,
    COLLECTABLE_BLOCKS,
    N_ACTIONS,
    REQUIRED_TOOL,
    Action,
    Block,
)
from goalmisgen.craftax.routes import canonical_moves
from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.level import Level
from goalmisgen.envs.solver import MOVES, TIE_TOLERANCE, LevelSolution, walls_blocking_other_objectives
from goalmisgen.offline.demonstrations import TASK_FILE, register_task
from goalmisgen.offline.demos import DEFAULT_MAX_ACTIONS, NO_ACTION, DemoSet
from goalmisgen.parallel import worker_pool

TASK = "craftax-ore"
"""Registry name written to ``task.json``."""

ROUTE_FIELDS: tuple[str, ...] = ("actions", "lengths", "distances", "target", "ambiguous", "utility_margin")
"""Arrays the engine expert adds beside the maze set's, saved as ``craftax_<name>.npy``."""

ROUTE_PREFIX = "craftax_"

MOVE_TO_ACTION: tuple[int, ...] = (int(Action.UP), int(Action.DOWN), int(Action.LEFT), int(Action.RIGHT))
"""Engine action for each maze move, in :data:`~goalmisgen.envs.solver.MOVES` order."""
assert MOVES == ((-1, 0), (1, 0), (0, -1), (0, 1)), "MOVE_TO_ACTION is spelled for this MOVES order"


def solution_from_costs(
    level, costs: Sequence[int | None], step_penalty: float, step_limit: int | None = None
) -> LevelSolution:
    """The solver's verdict for given per-objective action costs: ``value - step_penalty x cost``.

    ``level`` only needs ``objectives`` with values. Same tie and margin rules
    as :func:`~goalmisgen.envs.solver.solve`, so the two agree when the costs
    are the maze distances.
    """
    costs = tuple(costs)
    if step_limit is not None:
        costs = tuple(None if c is None or c > step_limit else c for c in costs)
    utilities = [None if c is None else objective.value - step_penalty * c for objective, c in zip(level.objectives, costs)]
    reachable = [(i, u) for i, u in enumerate(utilities) if u is not None]
    if not reachable:
        raise ValueError("no objective is reachable" + ("" if step_limit is None else f" within {step_limit} actions"))
    best = max(u for _, u in reachable)
    optimal = tuple(i for i, u in reachable if abs(u - best) <= TIE_TOLERANCE)
    others = [u for i, u in reachable if i != optimal[0]]
    margin = float("inf") if not others else best - max(others)
    return LevelSolution(
        distances=costs, utilities=tuple(utilities), optimal_index=optimal[0], optimal_indices=optimal, utility_margin=margin
    )


@dataclasses.dataclass(frozen=True)
class CraftaxTask:
    """How a maze level becomes a Craftax world, and a maze route a Craftax route."""

    kinds: tuple[int, ...] = (int(Block.COAL), int(Block.DIAMOND))
    """The ore tile standing in for each feature id (colour). Distinct, collectable."""

    start_direction: int = int(Action.DOWN)
    """The player's facing at t=0. Fixed, so it carries no information."""

    def __post_init__(self) -> None:
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError(f"kinds must be distinct, got {self.kinds}")
        for kind in self.kinds:
            if Block(kind) not in COLLECTABLE_BLOCKS:
                raise ValueError(f"{Block(kind).name} cannot be collected by DO")
            if Block(kind) == Block.TREE:
                raise ValueError("TREE turns into GRASS when mined and stays walkable; use an ore")
        if Action(self.start_direction) not in (Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN):
            raise ValueError(f"start_direction must be a move, got {Action(self.start_direction).name}")

    @property
    def n_features(self) -> int:
        return len(self.kinds)

    @property
    def interactable(self) -> tuple[int, ...]:
        """Solid blocks a blocked move may turn to face without being illegal: the ores."""
        return self.kinds

    @property
    def tools(self) -> tuple[str, ...]:
        """The pickaxes the player must hold to mine every kind."""
        return tuple(sorted({REQUIRED_TOOL[Block(k)] for k in self.kinds} - {None}))

    def observe(
        self, tiles: np.ndarray, agent: tuple[int, int], feature_values: Sequence[float], hide_values: bool = False
    ) -> np.ndarray:
        """The model's observation from a tile grid: the maze's channel layout.

        Wall = bedrock; agent one-hot; one channel per kind; and, unless hidden,
        ``feature_values[k]`` on the cells of kind ``k``.
        """
        from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL

        tiles = np.asarray(tiles)
        n_channels = FIRST_FEATURE_CHANNEL + self.n_features + (0 if hide_values else 1)
        observation = np.zeros(tiles.shape + (n_channels,), dtype=np.float32)
        observation[..., WALL_CHANNEL] = tiles == int(BEDROCK)
        observation[agent[0], agent[1], AGENT_CHANNEL] = 1.0
        for k, kind in enumerate(self.kinds):
            mask = tiles == kind
            observation[mask, FIRST_FEATURE_CHANNEL + k] = 1.0
            if not hide_values:
                observation[mask, FIRST_FEATURE_CHANNEL + self.n_features] = float(feature_values[k])
        return observation

    def tiles(self, level: Level) -> np.ndarray:
        """``(H, W)`` int32 block ids: walls bedrock, free grass, objectives their ore."""
        tiles = np.where(level.walls, int(BEDROCK), int(Block.GRASS)).astype(np.int32)
        for objective in level.objectives:
            tiles[objective.position] = self.kinds[objective.feature_id]
        return tiles

    def route(self, moves: np.ndarray, width: int | None = None) -> np.ndarray:
        """Maze moves (``NO_ACTION`` padded) -> engine actions, ``width`` wide, DO appended.

        The maze path's last move steps onto the objective; in the engine it is
        the turn to face it. After a straight approach the player already faces
        the ore, so that move is dropped rather than wasted. ``width`` defaults
        to one more than the moves, the most a route can need.
        """
        moves = np.asarray(moves)
        valid = moves >= 0
        length = int(valid.sum())
        kept = moves[:length].tolist()
        if len(kept) >= 2 and kept[-1] == kept[-2]:
            kept = kept[:-1]
        width = moves.shape[0] + 1 if width is None else width
        out = np.full(width, NO_ACTION, dtype=np.int32)
        out[: len(kept)] = np.asarray(MOVE_TO_ACTION, dtype=np.int32)[kept]
        out[len(kept)] = int(Action.DO)
        return out

    def route_to(self, level: Level, index: int, width: int | None = None) -> np.ndarray | None:
        """The engine route to objective ``index``, or ``None`` if it is unreachable.

        A shortest path under the canonical tie-break of
        :mod:`goalmisgen.craftax.routes`, routing around the other objectives
        as the maze solver does; same length as the maze's path, so the
        solver's distances and choice are unchanged.
        """
        moves = canonical_moves(
            walls_blocking_other_objectives(level, index), level.agent_start, level.objectives[index].position
        )
        if moves is None:
            return None
        return self.route(np.asarray(moves, dtype=np.int32), width)

    def costs(self, level: Level) -> tuple[int | None, ...]:
        """Actions the engine route to each objective takes; ``None`` if blocked."""
        out = []
        for index in range(level.n_objectives):
            route = self.route_to(level, index)
            out.append(None if route is None else int((route >= 0).sum()))
        return tuple(out)

    def solution(self, level: Level, step_penalty: float, step_limit: int | None = None) -> LevelSolution:
        """``solve()`` with engine costs in place of maze distances."""
        return solution_from_costs(level, self.costs(level), step_penalty, step_limit)

    def to_json(self) -> dict:
        return {"kinds": list(self.kinds), "start_direction": self.start_direction}

    @classmethod
    def from_json(cls, data: dict) -> "CraftaxTask":
        return cls(kinds=tuple(int(k) for k in data["kinds"]), start_direction=int(data["start_direction"]))


@dataclasses.dataclass(frozen=True)
class CraftaxDemoSet:
    """A maze demonstration set executed as Craftax routes; see the module docstring.

    Implements :class:`~goalmisgen.offline.demonstrations.Demonstrations`.
    """

    inner: DemoSet
    task: CraftaxTask
    actions: np.ndarray  # (N, max_actions) int8 engine actions, NO_ACTION padded
    lengths: np.ndarray  # (N,) int16
    distances: np.ndarray  # (N, K) int16 engine route cost per objective, -1 if blocked
    target: np.ndarray  # (N,) int8
    ambiguous: np.ndarray  # (N,) bool
    utility_margin: np.ndarray  # (N,) float32
    path: pathlib.Path | None = None

    # --- shape ---------------------------------------------------------------
    @property
    def size(self) -> int:
        return self.inner.size

    @property
    def hide_values(self) -> bool:
        return self.inner.hide_values

    @property
    def n_channels(self) -> int:
        return self.inner.n_channels

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def max_actions(self) -> int:
        return self.actions.shape[1]

    @property
    def move_actions(self) -> tuple[int, ...]:
        return MOVE_TO_ACTION

    @property
    def rho(self) -> float:
        return self.inner.rho

    @property
    def meta(self) -> dict:
        """The inner header with this task's own limits, plus the task."""
        return {
            **self.inner.meta,
            "step_limit": int(self.inner.meta["step_limit"]) + 1,
            "max_actions": self.max_actions,
            "task": {"task": TASK, **self.task.to_json()},
        }
        # step_limit: the maze expert was given one fewer so that every engine route fits.

    def __len__(self) -> int:
        return len(self.inner)

    # --- ground truth ----------------------------------------------------------
    @property
    def level_index(self) -> np.ndarray:
        return self.inner.level_index

    @property
    def values(self) -> np.ndarray:
        return self.inner.values

    @property
    def feature_ids(self) -> np.ndarray:
        return self.inner.feature_ids

    @property
    def agent(self) -> np.ndarray:
        return self.inner.agent

    @property
    def positions(self) -> np.ndarray:
        return self.inner.positions

    # --- what the model sees and emits ----------------------------------------
    def observations(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        return self.inner.observations(indices)

    def routes(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        return np.asarray(self.actions[np.asarray(indices)]).astype(np.int32)

    def level(self, index: int) -> Level:
        return self.inner.level(index)

    def replay(self, index: int, actions: Sequence[int], emitted_eos: bool = True) -> dict:
        from goalmisgen.craftax import engine  # JAX; kept out of the generation path

        return engine.replay(
            self.level(index),
            self.task,
            np.asarray(actions),
            float(self.inner.meta["step_penalty"]),
            int(self.meta["step_limit"]),
            emitted_eos,
        )

    def replay_many(self, indices: Sequence[int], actions: np.ndarray, emitted_eos: Sequence[bool]) -> list[dict]:
        """All routes of a batch in one engine rollout; what ``decode.replay_all`` prefers."""
        from goalmisgen.craftax import engine

        return engine.replay_batch(
            [self.level(int(index)) for index in indices],
            self.task,
            np.asarray(actions),
            float(self.inner.meta["step_penalty"]),
            int(self.meta["step_limit"]),
            list(emitted_eos),
        )

    # --- views ----------------------------------------------------------------
    def subset(self, indices: np.ndarray | Sequence[int]) -> "CraftaxDemoSet":
        indices = np.asarray(indices)
        own = {name: np.asarray(getattr(self, name)[indices]) for name in ROUTE_FIELDS}
        return dataclasses.replace(self, inner=self.inner.subset(indices), path=None, **own)

    def with_hidden_values(self, hide: bool = True) -> "CraftaxDemoSet":
        return dataclasses.replace(self, inner=self.inner.with_hidden_values(hide))

    def with_values(self, values: np.ndarray) -> "CraftaxDemoSet":
        return dataclasses.replace(self, inner=self.inner.with_values(values))

    def with_feature_ids(self, feature_ids: np.ndarray) -> "CraftaxDemoSet":
        return dataclasses.replace(self, inner=self.inner.with_feature_ids(feature_ids))

    # --- persistence ----------------------------------------------------------
    def save(self, path: str | pathlib.Path) -> None:
        directory = pathlib.Path(path)
        self.inner.save(directory)
        for name in ROUTE_FIELDS:
            np.save(directory / f"{ROUTE_PREFIX}{name}.npy", np.asarray(getattr(self, name)))
        (directory / TASK_FILE).write_text(json.dumps({"task": TASK, **self.task.to_json()}, indent=2))

    @classmethod
    def load(cls, path: str | pathlib.Path, mmap: bool = True, hide_values: bool = False) -> "CraftaxDemoSet":
        directory = pathlib.Path(path)
        marker = json.loads((directory / TASK_FILE).read_text())
        if marker.get("task") != TASK:
            raise ValueError(f"{directory} holds task {marker.get('task')!r}, not {TASK!r}")
        inner = DemoSet.load(directory, mmap=mmap, hide_values=hide_values)
        own = {
            name: np.load(directory / f"{ROUTE_PREFIX}{name}.npy", mmap_mode="r" if mmap else None) for name in ROUTE_FIELDS
        }
        return cls(inner=inner, task=CraftaxTask.from_json(marker), path=directory, **own)

    @classmethod
    def generate(
        cls,
        dataset: LevelDataset,
        indices: np.ndarray,
        rho: float,
        task: CraftaxTask | None = None,
        colour_keyed: bool = False,
        seed: int = 0,
        step_penalty: float = 0.05,
        step_limit: int = 120,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        workers: int = 1,
        chunk_size: int = 5_000,
        split: str | None = None,
    ) -> "CraftaxDemoSet":
        """Demonstrate levels for the engine.

        The maze set supplies the levels, colours and observation (its expert is
        given one fewer step and action, so that any engine route fits); the
        engine routes, costs and choices are planned here, level by level, in
        the same spawned pool.
        """
        task = task or CraftaxTask()
        inner = DemoSet.generate(
            dataset,
            indices,
            rho=rho,
            colour_keyed=colour_keyed,
            seed=seed,
            step_penalty=step_penalty,
            step_limit=step_limit - 1,
            max_actions=max_actions - 1,
            workers=workers,
            chunk_size=chunk_size,
            split=split,
        )
        rows = np.arange(len(inner))
        chunks = [rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]
        tasks = [(inner.subset(chunk), task, step_penalty, step_limit, max_actions) for chunk in chunks]
        if workers > 1 and len(tasks) > 1:
            with worker_pool(workers) as pool:
                blocks = pool.starmap(plan_block, tasks)
        else:
            blocks = [plan_block(*args) for args in tasks]
        own = {name: np.concatenate([block[name] for block in blocks]) for name in ROUTE_FIELDS}
        return cls(inner=inner, task=task, **own)


def plan_block(inner: DemoSet, task: CraftaxTask, step_penalty: float, step_limit: int, max_actions: int) -> dict:
    """Engine routes and choices for one chunk of maze levels. Safe in a worker process."""
    count = len(inner)
    n_objectives = inner.n_objectives
    out = {
        "actions": np.full((count, max_actions), NO_ACTION, dtype=np.int8),
        "lengths": np.empty(count, dtype=np.int16),
        "distances": np.full((count, n_objectives), -1, dtype=np.int16),
        "target": np.empty(count, dtype=np.int8),
        "ambiguous": np.empty(count, dtype=np.bool_),
        "utility_margin": np.empty(count, dtype=np.float32),
    }
    for row in range(count):
        level = inner.level(row)
        solution = task.solution(level, step_penalty, step_limit)
        route = task.route_to(level, solution.optimal_index)
        assert route is not None
        length = int((route >= 0).sum())
        if length > max_actions:
            raise ValueError(f"level {inner.level_index[row]} needs {length} actions but max_actions={max_actions}")
        out["actions"][row, :length] = route[:length]
        out["lengths"][row] = length
        out["distances"][row] = [-1 if c is None else c for c in solution.distances]
        out["target"][row] = solution.optimal_index
        out["ambiguous"][row] = solution.is_ambiguous
        out["utility_margin"][row] = solution.utility_margin if np.isfinite(solution.utility_margin) else np.inf
    return out


@register_task(TASK)
def _load(path: pathlib.Path, mmap: bool, hide_values: bool) -> CraftaxDemoSet:
    return CraftaxDemoSet.load(path, mmap=mmap, hide_values=hide_values)
