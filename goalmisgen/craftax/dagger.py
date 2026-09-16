"""DAgger for the crafting task: train on the states the policy actually visits.

The expert never errs, so the suffix pool holds only states on expert routes,
and a closed-loop policy that strays - crafts a step early, wanders - meets
states no training example covers and, being deterministic, can repeat a
no-op until the step cap. The remedy is Ross et al.'s: roll the policy out,
label every state it visits with what the expert would do *from there*
(:meth:`CraftTask.plans_from` gives that for any state), and train on those
pairs alongside the expert's own. :class:`StateDemoSet` is that store, a
:class:`~goalmisgen.offline.demonstrations.Demonstrations` over arbitrary
states rather than fields, and :class:`MixedDemoSet` lets the trainer draw
from it and the suffix pool together.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from goalmisgen.craftax import simulate
from goalmisgen.craftax.craft import MOVE_TO_ACTION, CraftDemoSet, CraftTask
from goalmisgen.craftax.demos import solution_from_costs
from goalmisgen.offline.demonstrations import TASK_FILE, register_task
from goalmisgen.offline.demos import NO_ACTION

TASK = "craftax-craft-states"

ARRAY_FIELDS: tuple[str, ...] = (
    "field_index",
    "tiles",
    "position",
    "facing",
    "inventory",
    "feature_values",
    "values",
    "feature_ids",
    "actions",
    "lengths",
    "distances",
    "target",
    "ambiguous",
    "utility_margin",
)


@dataclasses.dataclass(frozen=True)
class StateDemoSet:
    """(state, remaining expert route) pairs over arbitrary states of known fields."""

    field_index: np.ndarray  # (N,) index into the source CraftDemoSet
    tiles: np.ndarray  # (N, S, S) uint8
    position: np.ndarray  # (N, 2) uint8
    facing: np.ndarray  # (N,) int8
    inventory: np.ndarray  # (N, 4) int8
    feature_values: np.ndarray  # (N, K) float32
    values: np.ndarray  # (N, K) per objective, as the field's
    feature_ids: np.ndarray  # (N, K)
    actions: np.ndarray  # (N, T) int8
    lengths: np.ndarray  # (N,) int16
    distances: np.ndarray  # (N, K) plan cost per objective from this state, -1 if none
    target: np.ndarray  # (N,) int8
    ambiguous: np.ndarray  # (N,) bool
    utility_margin: np.ndarray  # (N,) float32
    size: int
    meta: dict
    task: CraftTask
    path: pathlib.Path | None = None
    hide_values: bool = False

    def __len__(self) -> int:
        return len(self.field_index)

    @property
    def n_channels(self) -> int:
        return self.task.n_channels(self.hide_values)

    @property
    def n_actions(self) -> int:
        return 17

    @property
    def max_actions(self) -> int:
        return self.actions.shape[1]

    @property
    def move_actions(self) -> tuple[int, ...]:
        return MOVE_TO_ACTION

    @property
    def rho(self) -> float:
        return float(self.meta["rho"])

    @property
    def level_index(self) -> np.ndarray:
        return np.asarray(self.field_index)

    @property
    def agent(self) -> np.ndarray:
        return np.asarray(self.position)

    @property
    def positions(self) -> np.ndarray:
        raise NotImplementedError("objective positions belong to the source fields; use the source set")

    def state(self, index: int) -> simulate.State:
        return simulate.State(
            np.asarray(self.tiles[index], dtype=np.int32),
            tuple(int(v) for v in self.position[index]),
            int(self.facing[index]),
            tuple(int(v) for v in self.inventory[index]),
        )

    def observations(self, indices) -> np.ndarray:
        indices = np.asarray(indices)
        return self.task.observe_batch(
            np.asarray(self.tiles[indices], dtype=np.int32),
            np.asarray(self.position[indices]),
            np.asarray(self.feature_values[indices]),
            self.hide_values,
            np.asarray(self.facing[indices]),
            np.asarray(self.inventory[indices]),
        )

    def routes(self, indices) -> np.ndarray:
        return np.asarray(self.actions[np.asarray(indices)]).astype(np.int32)

    def level(self, index: int):
        raise NotImplementedError("a state set trains; evaluate on the source CraftDemoSet")

    def replay(self, index: int, actions, emitted_eos: bool = True) -> dict:
        raise NotImplementedError("a state set trains; evaluate on the source CraftDemoSet")

    def subset(self, indices) -> "StateDemoSet":
        indices = np.asarray(indices)
        return dataclasses.replace(self, **{n: np.asarray(getattr(self, n)[indices]) for n in ARRAY_FIELDS}, path=None)

    def with_hidden_values(self, hide: bool = True) -> "StateDemoSet":
        return dataclasses.replace(self, hide_values=hide)

    def with_values(self, values):
        raise NotImplementedError

    def with_feature_ids(self, feature_ids):
        raise NotImplementedError

    def save(self, path) -> None:
        directory = pathlib.Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ARRAY_FIELDS:
            np.save(directory / f"{name}.npy", np.asarray(getattr(self, name)))
        (directory / "meta.json").write_text(json.dumps({**self.meta, "size": self.size, "n": len(self)}, indent=2))
        (directory / TASK_FILE).write_text(json.dumps({"task": TASK, **self.task.to_json()}, indent=2))

    @classmethod
    def load(cls, path, mmap: bool = True, hide_values: bool = False) -> "StateDemoSet":
        directory = pathlib.Path(path)
        meta = json.loads((directory / "meta.json").read_text())
        marker = json.loads((directory / TASK_FILE).read_text())
        arrays = {n: np.load(directory / f"{n}.npy", mmap_mode="r" if mmap else None) for n in ARRAY_FIELDS}
        return cls(
            **arrays,
            size=int(meta["size"]),
            meta=meta,
            task=CraftTask.from_json(marker),
            path=directory,
            hide_values=hide_values,
        )


@register_task(TASK)
def _load(path: pathlib.Path, mmap: bool, hide_values: bool) -> StateDemoSet:
    return StateDemoSet.load(path, mmap=mmap, hide_values=hide_values)


def label(task: CraftTask, field, state: simulate.State, step_penalty: float, step_limit: int, max_actions: int):
    """The expert's route from ``state``, or ``None`` if no objective can be completed from it."""
    plans = task.plans_from(state, field)
    if all(p is None for p in plans):
        return None
    solution = solution_from_costs(field, [None if p is None else p.cost for p in plans], step_penalty, step_limit)
    plan = plans[solution.optimal_index]
    if plan is None or plan.cost > max_actions:
        return None
    return plan, solution


def collect(
    model,
    params,
    demos: CraftDemoSet,
    indices: np.ndarray,
    seed: int = 0,
    avoid_ineffective_repeat: bool = False,
    states_per_route: int | None = None,
    rng: np.random.Generator | None = None,
) -> StateDemoSet:
    """Roll the policy out on ``indices`` of ``demos`` and label every visited state.

    ``states_per_route`` subsamples the visited states of each route at random
    (all of them by default), which keeps the labelling cost proportional.
    """
    from goalmisgen.craftax import closed_loop

    task = demos.task
    indices = np.asarray(indices)
    step_penalty, step_limit, max_actions = float(demos.meta["step_penalty"]), int(demos.meta["step_limit"]), demos.max_actions
    record: list = []
    closed_loop.rollout(
        model, params, demos, indices, seed=seed, avoid_ineffective_repeat=avoid_ineffective_repeat, record=record
    )
    rng = rng or np.random.default_rng(seed)
    per_field: dict[int, list] = {}
    for rows, tiles, positions, facings, inventories, active in record:
        for k, field_index in enumerate(rows):
            if active[k]:
                per_field.setdefault(int(field_index), []).append((tiles[k], positions[k], facings[k], inventories[k]))
    out = {name: [] for name in ARRAY_FIELDS}
    n_objectives = demos.n_objectives
    for field_index, visited in per_field.items():
        if states_per_route is not None and len(visited) > states_per_route:
            keep = sorted(rng.choice(len(visited), states_per_route, replace=False))
            visited = [visited[i] for i in keep]
        field = demos.level(field_index)
        feature_values = demos.feature_values([field_index])[0]
        for tiles, position, facing, inventory in visited:
            state = simulate.State(
                np.asarray(tiles, dtype=np.int32),
                tuple(int(v) for v in position),
                int(facing),
                tuple(int(v) for v in inventory),
            )
            labelled = label(task, field, state, step_penalty, step_limit, max_actions)
            if labelled is None:
                continue
            plan, solution = labelled
            route = np.full(max_actions, NO_ACTION, dtype=np.int8)
            route[: plan.cost] = plan.actions
            out["field_index"].append(field_index)
            out["tiles"].append(np.asarray(tiles, dtype=np.uint8))
            out["position"].append(np.asarray(position, dtype=np.uint8))
            out["facing"].append(np.int8(facing))
            out["inventory"].append(np.asarray(inventory, dtype=np.int8))
            out["feature_values"].append(feature_values.astype(np.float32))
            out["values"].append(np.asarray(demos.values[field_index]))
            out["feature_ids"].append(np.asarray(demos.feature_ids[field_index]))
            out["actions"].append(route)
            out["lengths"].append(np.int16(plan.cost))
            out["distances"].append(np.asarray([-1 if c is None else c for c in solution.distances], dtype=np.int16))
            out["target"].append(np.int8(solution.optimal_index))
            out["ambiguous"].append(solution.is_ambiguous)
            out["utility_margin"].append(
                np.float32(solution.utility_margin if np.isfinite(solution.utility_margin) else np.inf)
            )
    arrays = {name: np.stack(v) if v else np.zeros((0,), dtype=np.int8) for name, v in out.items()}
    meta = {
        **{k: demos.meta[k] for k in ("rho", "values", "step_penalty", "step_limit", "max_actions", "source_fingerprint")},
        "task": {"task": TASK, **task.to_json()},
        "source": None if demos.path is None else str(demos.path),
        "dagger": {
            "fields": int(len(indices)),
            "seed": int(seed),
            "guard": bool(avoid_ineffective_repeat),
            "states_per_route": states_per_route,
        },
        "n": int(len(arrays["field_index"])),
    }
    del n_objectives
    return StateDemoSet(**arrays, size=demos.size, meta=meta, task=task, hide_values=demos.hide_values)


@dataclasses.dataclass(frozen=True)
class MixedDemoSet:
    """Two demonstration sets drawn from as one, for the trainer."""

    parts: tuple

    def __post_init__(self) -> None:
        sizes = {p.size for p in self.parts}
        channels = {p.n_channels for p in self.parts}
        widths = {p.max_actions for p in self.parts}
        if len(sizes) != 1 or len(channels) != 1 or len(widths) != 1:
            raise ValueError(f"parts disagree in shape: sizes {sizes}, channels {channels}, max_actions {widths}")
        object.__setattr__(self, "_offsets", np.cumsum([0] + [len(p) for p in self.parts]))

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _split(self, indices):
        indices = np.asarray(indices)
        part = np.searchsorted(self._offsets, indices, side="right") - 1
        return [(k, np.nonzero(part == k)[0], indices[part == k] - self._offsets[k]) for k in range(len(self.parts))]

    def _gather(self, indices, method: str) -> np.ndarray:
        indices = np.asarray(indices)
        pieces = []
        order = []
        for k, rows, local in self._split(indices):
            if len(local):
                pieces.append(getattr(self.parts[k], method)(local))
                order.append(rows)
        out = np.concatenate(pieces)
        result = np.empty_like(out)
        result[np.concatenate(order)] = out
        return result

    def observations(self, indices) -> np.ndarray:
        return self._gather(indices, "observations")

    def routes(self, indices) -> np.ndarray:
        return self._gather(indices, "routes")

    @property
    def lengths(self) -> np.ndarray:
        return np.concatenate([np.asarray(p.lengths) for p in self.parts])

    @property
    def level_index(self) -> np.ndarray:
        return np.concatenate([np.asarray(p.level_index) for p in self.parts])

    def __getattr__(self, name):  # shape and header attributes: the first part's
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.parts[0], name)

    @property
    def meta(self) -> dict:
        return {**self.parts[0].meta, "mixed": [{"n": len(p), "task": p.meta.get("task")} for p in self.parts]}

    @property
    def path(self):
        return self.parts[0].path
