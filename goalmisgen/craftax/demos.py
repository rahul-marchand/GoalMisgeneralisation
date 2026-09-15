"""The maze's demonstrations, executed as Craftax routes.

A stage-2 Craftax world *is* a maze level rendered into the engine (see
:mod:`goalmisgen.craftax.levels`), and the expert route is the maze route plus
one DO: the maze path's last move steps onto the objective, but in the engine
the ore is solid, so that move only turns the player to face it, and DO then
mines it. So a Craftax demonstration set is a :class:`~goalmisgen.offline.demos.DemoSet`
seen through a :class:`CraftaxTask`: the same arrays, the same observation, the
routes re-spelled in the engine's 17-action vocabulary with a DO appended, and
every distance one action longer. Nothing is stored twice, and a set at rho
here is paired level for level with the maze's at the same rho.

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
    COLLECTABLE_BLOCKS,
    N_ACTIONS,
    REQUIRED_TOOL,
    Action,
    Block,
)
from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.level import Level
from goalmisgen.envs.solver import MOVES
from goalmisgen.offline.demonstrations import TASK_FILE, register_task
from goalmisgen.offline.demos import DEFAULT_MAX_ACTIONS, NO_ACTION, DemoSet

TASK = "craftax-ore"
"""Registry name written to ``task.json``."""

MOVE_TO_ACTION: tuple[int, ...] = (int(Action.UP), int(Action.DOWN), int(Action.LEFT), int(Action.RIGHT))
"""Engine action for each maze move, in :data:`~goalmisgen.envs.solver.MOVES` order."""
assert MOVES == ((-1, 0), (1, 0), (0, -1), (0, 1)), "MOVE_TO_ACTION is spelled for this MOVES order"


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
    def tools(self) -> tuple[str, ...]:
        """The pickaxes the player must hold to mine every kind."""
        return tuple(sorted({REQUIRED_TOOL[Block(k)] for k in self.kinds} - {None}))

    def tiles(self, level: Level) -> np.ndarray:
        """``(H, W)`` int32 block ids: walls stone, free grass, objectives their ore."""
        tiles = np.where(level.walls, int(Block.STONE), int(Block.GRASS)).astype(np.int32)
        for objective in level.objectives:
            tiles[objective.position] = self.kinds[objective.feature_id]
        return tiles

    def route(self, moves: np.ndarray) -> np.ndarray:
        """Maze moves ``(T,)`` (``NO_ACTION`` padded) -> engine actions ``(T + 1,)``, DO appended."""
        moves = np.asarray(moves)
        out = np.full(moves.shape[0] + 1, NO_ACTION, dtype=np.int32)
        valid = moves >= 0
        length = int(valid.sum())
        out[:length] = np.asarray(MOVE_TO_ACTION, dtype=np.int32)[moves[:length]]
        out[length] = int(Action.DO)
        return out

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
    task: CraftaxTask = dataclasses.field(default_factory=CraftaxTask)
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
        return self.inner.max_actions + 1

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
    def distances(self) -> np.ndarray:
        """Maze distances plus the DO, where reachable."""
        distances = np.asarray(self.inner.distances)
        return np.where(distances >= 0, distances + 1, distances)

    @property
    def feature_ids(self) -> np.ndarray:
        return self.inner.feature_ids

    @property
    def target(self) -> np.ndarray:
        return self.inner.target

    @property
    def ambiguous(self) -> np.ndarray:
        return self.inner.ambiguous

    @property
    def utility_margin(self) -> np.ndarray:
        return self.inner.utility_margin

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.inner.lengths) + 1

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
        return np.stack([self.task.route(moves) for moves in self.inner.routes(indices)])

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
        return dataclasses.replace(self, inner=self.inner.subset(indices), path=None)

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
        (directory / TASK_FILE).write_text(json.dumps({"task": TASK, **self.task.to_json()}, indent=2))

    @classmethod
    def load(cls, path: str | pathlib.Path, mmap: bool = True, hide_values: bool = False) -> "CraftaxDemoSet":
        directory = pathlib.Path(path)
        marker = json.loads((directory / TASK_FILE).read_text())
        if marker.get("task") != TASK:
            raise ValueError(f"{directory} holds task {marker.get('task')!r}, not {TASK!r}")
        inner = DemoSet.load(directory, mmap=mmap, hide_values=hide_values)
        return cls(inner=inner, task=CraftaxTask.from_json(marker), path=directory)

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
        """Demonstrate levels for the engine: maze routes with one action to spare.

        ``step_limit`` and ``max_actions`` are the engine's; the maze expert is
        given one fewer of each so that its route plus the DO fits both.
        """
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
        return cls(inner=inner, task=task or CraftaxTask())


@register_task(TASK)
def _load(path: pathlib.Path, mmap: bool, hide_values: bool) -> CraftaxDemoSet:
    return CraftaxDemoSet.load(path, mmap=mmap, hide_values=hide_values)
