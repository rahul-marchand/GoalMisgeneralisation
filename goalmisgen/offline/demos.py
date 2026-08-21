"""Expert demonstrations: the training data of the offline experiment.

A demonstration is one level, with its colours assigned at some correlation,
and the route a BFS-optimal agent walks on it under the task's utility
``value - step_penalty * distance``. The route is computed by the same solver
that scores every agent in the project, so the expert is exactly the reference
the behavioural metrics already measure against.

The correlation lives in the *data*, not in a reward: at rho=1.0 colour 0
always marks the richer objective in every demonstration, and a model that
imitates those demonstrations is free to learn "go to colour 0" instead of "go
to the better trade of value against distance". That is the misgeneralisation
under study, arrived at by imitation rather than by reinforcement.

Storage mirrors :class:`~goalmisgen.envs.dataset.LevelDataset`: a directory
of plain ``.npy`` arrays plus a JSON header, self-contained so that training
needs neither the source level dataset nor its fingerprint check. The source
and its fingerprint are recorded in the header for provenance.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import pathlib
from typing import Sequence

import numpy as np

from goalmisgen.envs.dataset import LevelDataset, source_fingerprint
from goalmisgen.envs.features import CorrelatedFeatures
from goalmisgen.envs.level import Level
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL
from goalmisgen.envs.solver import MOVES, path_to_objective, solve

ARRAY_FIELDS: tuple[str, ...] = (
    "level_index",
    "walls_packed",
    "sizes",
    "agent",
    "positions",
    "values",
    "distances",
    "feature_ids",
    "actions",
    "lengths",
    "target",
    "ambiguous",
    "utility_margin",
)
"""Arrays persisted to disk, one ``.npy`` each."""

NO_ACTION = -1
"""Padding in the ``actions`` array past the end of a route."""

DEFAULT_MAX_ACTIONS = 64
"""Longest route a demonstration may hold.

An 11x11 perfect maze has at most 61 free cells, so no expert route exceeds 60
steps. Generation refuses a route that does not fit rather than truncating it,
because a truncated route would teach the model to stop short of the goal.
"""

_MOVE_INDEX = {move: index for index, move in enumerate(MOVES)}


@dataclasses.dataclass(frozen=True)
class DemoSet:
    """Expert demonstrations on a fixed pool of levels, at one correlation."""

    level_index: np.ndarray  # (N,) int64 - index into the source LevelDataset
    walls_packed: np.ndarray  # (N, ceil(S*S/8)) uint8, padded to S with wall
    sizes: np.ndarray  # (N,) uint8 - true maze side
    agent: np.ndarray  # (N, 2) uint8
    positions: np.ndarray  # (N, K, 2) uint8
    values: np.ndarray  # (N, K) float64
    distances: np.ndarray  # (N, K) int16 - route length to each objective, -1 if blocked
    feature_ids: np.ndarray  # (N, K) int8 - the colours, assigned at `rho`
    actions: np.ndarray  # (N, max_actions) int8 - the expert's moves, NO_ACTION padded
    lengths: np.ndarray  # (N,) int16 - route length in moves
    target: np.ndarray  # (N,) int8 - the objective the expert walks to
    ambiguous: np.ndarray  # (N,) bool - exact utility tie, so either choice was optimal
    utility_margin: np.ndarray  # (N,) float32
    size: int
    """Padded side of every maze; the observation is ``size x size``."""

    meta: dict
    path: pathlib.Path | None = None
    hide_values: bool = False
    """Leave the value channel out of the observation.

    The twin of ``value_encoding="none"`` / ``colour_is_the_only_value_cue`` in
    the environment (``novalue11``): with no value channel the values are
    learned constants rather than inputs, so nothing forces the model to
    represent them, and a value-axis fine-tune has to move *weights* to move
    what an objective is worth. The arrays are untouched; only what
    :meth:`observations` builds from them changes.
    """

    def __len__(self) -> int:
        return len(self.level_index)

    @property
    def n_objectives(self) -> int:
        return self.positions.shape[1]

    @property
    def n_features(self) -> int:
        return self.n_objectives

    @property
    def n_channels(self) -> int:
        """Channels of the observation: wall, agent, one per colour, one value (unless hidden)."""
        return FIRST_FEATURE_CHANNEL + self.n_features + (0 if self.hide_values else 1)

    @property
    def max_actions(self) -> int:
        return self.actions.shape[1]

    @property
    def rho(self) -> float:
        return float(self.meta["rho"])

    # ------------------------------------------------------------------
    # Reading levels back
    # ------------------------------------------------------------------

    def walls(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        """Unpacked, padded wall grids, ``(B, size, size)`` bool."""
        packed = np.asarray(self.walls_packed[np.asarray(indices)])
        flat = np.unpackbits(packed, axis=1, count=self.size * self.size)
        return flat.reshape(len(packed), self.size, self.size).astype(np.bool_)

    def level(self, index: int) -> Level:
        """The level as the environment would see it, colours included."""
        from goalmisgen.envs.level import Objective

        size = int(self.sizes[index])
        walls = self.walls([index])[0][:size, :size]
        return Level(
            walls=walls,
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

    def observations(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        """Observations for a batch, ``(B, size, size, channels)`` float32.

        Vectorised twin of :meth:`ObservationEncoder.encode` with
        ``value_encoding="at_objective"`` (or ``"none"`` when ``hide_values``);
        ``tests/test_offline_demos.py`` asserts they agree cell for cell. Built
        here rather than stored because a float observation is twenty times the
        size of the level it encodes.
        """
        indices = np.asarray(indices)
        batch = len(indices)
        observation = np.zeros((batch, self.size, self.size, self.n_channels), dtype=np.float32)
        observation[..., WALL_CHANNEL] = self.walls(indices)

        rows = np.arange(batch)
        agent = self.agent[indices]
        observation[rows, agent[:, 0], agent[:, 1], AGENT_CHANNEL] = 1.0

        value_channel = FIRST_FEATURE_CHANNEL + self.n_features
        for k in range(self.n_objectives):
            r, c = self.positions[indices, k, 0], self.positions[indices, k, 1]
            observation[rows, r, c, FIRST_FEATURE_CHANNEL + self.feature_ids[indices, k]] = 1.0
            if not self.hide_values:
                observation[rows, r, c, value_channel] = self.values[indices, k]
        return observation

    def routes(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        """Expert actions, ``(B, max_actions)`` int, ``NO_ACTION`` padded."""
        return np.asarray(self.actions[np.asarray(indices)]).astype(np.int32)

    def subset(self, indices: np.ndarray | Sequence[int]) -> "DemoSet":
        indices = np.asarray(indices)
        arrays = {name: np.asarray(getattr(self, name)[indices]) for name in ARRAY_FIELDS}
        return DemoSet(**arrays, size=self.size, meta={**self.meta, "n": int(len(indices))}, hide_values=self.hide_values)

    def with_hidden_values(self, hide: bool = True) -> "DemoSet":
        """The same demonstrations, observed without (or with) the value channel."""
        return dataclasses.replace(self, hide_values=hide)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        directory = pathlib.Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ARRAY_FIELDS:
            np.save(directory / f"{name}.npy", np.asarray(getattr(self, name)))
        (directory / "meta.json").write_text(json.dumps({**self.meta, "size": self.size, "n": len(self)}, indent=2))

    @classmethod
    def load(cls, path: str | pathlib.Path, mmap: bool = True, hide_values: bool = False) -> "DemoSet":
        directory = pathlib.Path(path)
        meta = json.loads((directory / "meta.json").read_text())
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None) for name in ARRAY_FIELDS}
        return cls(**arrays, size=int(meta["size"]), meta=meta, path=directory, hide_values=hide_values)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        dataset: LevelDataset,
        indices: np.ndarray,
        rho: float,
        seed: int = 0,
        step_penalty: float = 0.05,
        step_limit: int = 120,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        workers: int = 1,
        chunk_size: int = 5_000,
        split: str | None = None,
    ) -> "DemoSet":
        """Demonstrate every level in ``indices`` at correlation ``rho``.

        Colours are drawn from an rng seeded by ``(seed, level index)``, so the
        same level demonstrated at two correlations differs only in colour -
        and at rho=0.5 the coin flip is the same coin for every rho, so the
        sets are paired level for level.
        """
        indices = np.asarray(indices, dtype=np.int64)
        chunks = [indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size)]
        tasks = [(dataset, chunk, rho, seed, step_penalty, step_limit, max_actions) for chunk in chunks]
        if workers > 1 and len(tasks) > 1:
            with mp.Pool(workers) as pool:
                blocks = pool.starmap(demonstrate_block, tasks)
        else:
            blocks = [demonstrate_block(*task) for task in tasks]

        arrays = {name: np.concatenate([block[name] for block in blocks]) for name in ARRAY_FIELDS}
        values = sorted({float(v) for v in np.unique(dataset.values)}, reverse=True)
        meta = {
            "source": None if dataset.path is None else str(dataset.path),
            "source_fingerprint": dataset.fingerprint,
            "code_fingerprint": source_fingerprint(),
            "split": split,
            "rho": float(rho),
            "seed": int(seed),
            "values": values,
            "step_penalty": float(step_penalty),
            "step_limit": int(step_limit),
            "max_actions": int(max_actions),
            "n": int(len(indices)),
        }
        return cls(**arrays, size=int(dataset.max_size), meta=meta)


def expert_route(level: Level, step_penalty: float, step_limit: int) -> tuple[int, tuple[int, ...], object]:
    """Which objective the expert takes, the moves it makes, and the solution.

    Ties go to the lowest index, as :func:`~goalmisgen.envs.solver.solve`
    breaks them; the level is flagged ambiguous so a scorer can exclude it.
    """
    solution = solve(level, step_penalty, step_limit=step_limit)
    target = solution.optimal_index
    path = path_to_objective(level, target)
    if path is None:  # pragma: no cover - solve() already established reachability
        raise RuntimeError("optimal objective is unreachable; the solver and the path search disagree")
    moves = tuple(_MOVE_INDEX[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:]))
    return target, moves, solution


def demonstrate_block(
    dataset: LevelDataset,
    indices: np.ndarray,
    rho: float,
    seed: int,
    step_penalty: float,
    step_limit: int,
    max_actions: int,
) -> dict[str, np.ndarray]:
    """Demonstrate one chunk of levels. Safe to run in a worker process."""
    count = len(indices)
    n_objectives = dataset.n_objectives
    scheme = CorrelatedFeatures(rho)

    out = {
        "level_index": np.asarray(indices, dtype=np.int64),
        "walls_packed": np.asarray(dataset.walls_packed[indices]),
        "sizes": np.asarray(dataset.sizes[indices]),
        "agent": np.asarray(dataset.agent[indices]),
        "positions": np.asarray(dataset.positions[indices]),
        "values": np.asarray(dataset.values[indices]),
        "distances": np.asarray(dataset.distances[indices]),
        "feature_ids": np.empty((count, n_objectives), dtype=np.int8),
        "actions": np.full((count, max_actions), NO_ACTION, dtype=np.int8),
        "lengths": np.empty(count, dtype=np.int16),
        "target": np.empty(count, dtype=np.int8),
        "ambiguous": np.empty(count, dtype=np.bool_),
        "utility_margin": np.empty(count, dtype=np.float32),
    }

    for row, index in enumerate(indices):
        values = tuple(float(value) for value in dataset.values[index])
        rng = np.random.default_rng([int(seed), int(index)])
        feature_ids = scheme.assign(values, rng)
        level = dataset.level(int(index), feature_ids)

        target, moves, solution = expert_route(level, step_penalty, step_limit)
        if len(moves) > max_actions:
            raise ValueError(
                f"level {index} needs {len(moves)} moves but max_actions={max_actions}; "
                "a truncated route would teach the model to stop short"
            )
        out["feature_ids"][row] = feature_ids
        out["actions"][row, : len(moves)] = moves
        out["lengths"][row] = len(moves)
        out["target"][row] = target
        out["ambiguous"][row] = solution.is_ambiguous
        out["utility_margin"][row] = solution.utility_margin if np.isfinite(solution.utility_margin) else np.inf
    return out
