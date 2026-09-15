"""What the offline pipeline needs from a demonstration set, whatever the task.

The trainer, decoder, axis fit and probes were written against the maze's
:class:`~goalmisgen.offline.demos.DemoSet`. A second task - the same trade-off
executed inside the Craftax engine, and later its crafting chains - has to
feed the same pipeline without forking it, so the surface those consumers
actually use is written down here as a protocol, and each task supplies an
implementation. Two rules keep the boundary honest:

- **Everything a consumer reads is on the protocol.** ``analysis.behaviour``
  reads outcome dicts with ``.get``, so a missing key degrades silently to
  ``nan`` rather than raising; the way to catch a task that forgot something
  is to enumerate the surface and test each implementation against it
  (``tests/test_demonstrations.py``).
- **The task executes its own routes.** :meth:`Demonstrations.replay` turns a
  route into the outcome dict - the maze walks its rules in numpy, Craftax
  steps the real engine - and nothing downstream of that dict knows which.

``observations`` is the observation *before the first action*, the prefix the
route model conditions on. A task whose observation changes along the route
(a local view) supplies :meth:`trajectory` as well; the prefix model does not
use it, a history model would.
"""

from __future__ import annotations

import json
import pathlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

TASK_FILE = "task.json"
"""Marker beside a demonstration set naming the task that executes its routes.

Absent for the maze, whose sets predate the protocol; present for every other
task, holding at least ``{"task": <name>}`` plus that task's own parameters.
"""


@runtime_checkable
class Demonstrations(Protocol):
    """Expert demonstrations on a fixed pool of levels, for one task."""

    # --- shape of the problem --------------------------------------------
    size: int
    """Padded side of the observation grid; the model's cell tokens are ``size**2``."""

    hide_values: bool
    path: pathlib.Path | None
    meta: dict
    """Provenance and the parameters a run needs to reload: at least
    ``step_penalty``, ``step_limit``, ``rho``, ``values`` and ``source_fingerprint``."""

    @property
    def n_channels(self) -> int:
        ...

    @property
    def n_actions(self) -> int:
        """Size of the model's action vocabulary, excluding EOS."""
        ...

    @property
    def max_actions(self) -> int:
        ...

    @property
    def move_actions(self) -> tuple[int, ...]:
        """Model action id of each maze move, in ``solver.MOVES`` order."""
        ...

    @property
    def rho(self) -> float:
        ...

    def __len__(self) -> int:
        ...

    # --- per-level ground truth, ``(N, K)`` over objectives or ``(N,)`` -----
    level_index: np.ndarray
    values: np.ndarray
    distances: np.ndarray
    """Route length to each objective in *this task's* actions, -1 if blocked."""
    feature_ids: np.ndarray
    target: np.ndarray
    ambiguous: np.ndarray
    utility_margin: np.ndarray
    lengths: np.ndarray
    agent: np.ndarray
    positions: np.ndarray

    # --- what the model sees and emits -------------------------------------
    def observations(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        """``(B, size, size, n_channels)`` float32, before the first action."""
        ...

    def routes(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        """``(B, max_actions)`` int model action ids, ``NO_ACTION`` padded."""
        ...

    def level(self, index: int):
        ...

    def replay(self, index: int, actions: Sequence[int], emitted_eos: bool = True) -> dict:
        """Execute ``actions`` on level ``index``; the outcome dict ``analysis.behaviour`` reads."""
        ...

    # --- views ------------------------------------------------------------
    def subset(self, indices: np.ndarray | Sequence[int]) -> "Demonstrations":
        ...

    def with_hidden_values(self, hide: bool = True) -> "Demonstrations":
        ...

    def with_values(self, values: np.ndarray) -> "Demonstrations":
        ...

    def with_feature_ids(self, feature_ids: np.ndarray) -> "Demonstrations":
        ...

    def save(self, path: str | pathlib.Path) -> None:
        ...


PROTOCOL_ATTRIBUTES: tuple[str, ...] = tuple(
    sorted(
        set(Demonstrations.__annotations__)
        | {
            name
            for name, value in vars(Demonstrations).items()
            if not name.startswith("_") and (callable(value) or isinstance(value, property))
        }
    )
)
"""Every name on the protocol, for the conformance test (3.11 has no ``__protocol_attrs__``)."""


def task_name(path: str | pathlib.Path) -> str | None:
    """The task recorded beside a saved demonstration set, or ``None`` for the maze."""
    marker = pathlib.Path(path) / TASK_FILE
    if not marker.exists():
        return None
    return str(json.loads(marker.read_text())["task"])


def load_demonstrations(path: str | pathlib.Path, mmap: bool = True, hide_values: bool = False) -> Demonstrations:
    """Load a saved demonstration set as whichever task wrote it.

    Same signature as :meth:`DemoSet.load`, which it replaces at the entry
    points that must accept any task. Task modules register themselves here
    rather than being imported eagerly: the Craftax loader is JAX-free, but a
    registry keeps this module from knowing every task by name.
    """
    from goalmisgen.offline.demos import DemoSet

    name = task_name(path)
    if name is None:
        return DemoSet.load(path, mmap=mmap, hide_values=hide_values)
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(f"{path} was written by task {name!r}, which no loader is registered for: {sorted(_LOADERS)}")
    return loader(pathlib.Path(path), mmap, hide_values)


_LOADERS: dict[str, object] = {}


def register_task(name: str):
    """Decorator registering ``loader(path, mmap, hide_values)`` for a task name."""

    def wrap(loader):
        if name in _LOADERS and _LOADERS[name] is not loader:
            raise ValueError(f"task {name!r} already has a loader")
        _LOADERS[name] = loader
        return loader

    return wrap
