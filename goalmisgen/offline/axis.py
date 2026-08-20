"""Reading a value-axis sweep of the route model off the volume.

The DRC campaign's loaders (``014``, ``015``) are written against cleanba
checkpoints; the route model saves flax params. What is shared is the
arithmetic - :mod:`goalmisgen.analysis.weights` - and the directory grammar -
:func:`goalmisgen.volume.arm_dirname`. This module supplies the rest: every
arm of one sweep as a flat weight diff from its base, keyed by the offset it
was trained at, and one behavioural measurement of any parameter vector.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
from jax.flatten_util import ravel_pytree

from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import RoutePrefixLM
from goalmisgen.offline.train import list_checkpoints, load_checkpoint, load_run_config
from goalmisgen.volume import parse_arm_dirname


@dataclasses.dataclass(frozen=True)
class Base:
    """A base run: its model, its final parameters flattened, and how to unflatten."""

    run_dir: pathlib.Path
    model: RoutePrefixLM
    params: dict
    flat: np.ndarray
    unravel: object
    step: int
    hide_values: bool
    values: tuple[float, ...]


def load_base(run_dir: pathlib.Path) -> Base:
    run_dir = pathlib.Path(run_dir)
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"{run_dir} has no checkpoints")
    step, directory = checkpoints[-1]
    model, params = load_checkpoint(directory)
    flat, unravel = ravel_pytree(params)
    config = load_run_config(run_dir)
    return Base(
        run_dir=run_dir,
        model=model,
        params=params,
        flat=np.asarray(flat, dtype=np.float64),
        unravel=unravel,
        step=step,
        hide_values=bool(config["demos"].get("hide_values", False)),
        values=tuple(float(v) for v in config["demos"]["values"]),
    )


def arm_dirs(run_dir: pathlib.Path, sweep: str, steps: int) -> dict[float, pathlib.Path]:
    """Finished arms of one sweep at one budget, keyed by offset."""
    root = pathlib.Path(run_dir) / "arms"
    if not root.is_dir():
        return {}
    found: dict[float, pathlib.Path] = {}
    for directory in sorted(root.iterdir()):
        parsed = parse_arm_dirname(directory.name)
        if parsed is None or parsed.sweep != sweep or parsed.steps != steps:
            continue
        if not (directory / "done.json").exists() or not list_checkpoints(directory):
            continue
        found[round(parsed.offset, 10)] = directory
    return found


def load_diffs(base: Base, arms: dict[float, pathlib.Path]) -> dict[float, np.ndarray]:
    """``theta_arm - theta_base`` for every arm, float64, keyed by offset."""
    diffs = {}
    for offset, directory in sorted(arms.items()):
        _, params = load_checkpoint(list_checkpoints(directory)[-1][1])
        flat, _ = ravel_pytree(params)
        diffs[offset] = np.asarray(flat, dtype=np.float64) - base.flat
    return diffs


def arm_params(directory: pathlib.Path):
    return load_checkpoint(list_checkpoints(directory)[-1][1])[1]


@dataclasses.dataclass(frozen=True)
class Measurement:
    """What one parameter vector does on the held-out levels."""

    indifference: float
    """Distance gap at which colour 0 (the richer objective at the base values) is taken half the time."""

    chose_optimal: float
    reached: float
    legal: float
    followed_f0: float

    def as_row(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def measure(base: Base, params, demos: DemoSet, indices: np.ndarray) -> Measurement:
    summary, _, _ = evaluate(base.model, params, demos, indices)
    b = summary.behaviour
    return Measurement(
        indifference=float(summary.indifference),
        chose_optimal=float(b.chose_optimal),
        reached=float(b.reached_objective),
        legal=float(summary.legal),
        followed_f0=float(b.followed_feature_zero),
    )


def measure_flat(base: Base, flat: np.ndarray, demos: DemoSet, indices: np.ndarray) -> Measurement:
    return measure(base, base.unravel(np.asarray(flat, dtype=np.float32)), demos, indices)


def expected_indifference(base_values: tuple[float, ...], objective: int, offset: float, step_penalty: float = 0.05) -> float:
    """The expert's exchange rate, in steps, once ``objective`` is moved by ``offset``.

    Positive means colour 0 (the richer objective at the base values) is worth
    that many extra steps; the sign convention matches :func:`measure`.
    """
    values = list(base_values)
    values[objective] += offset
    return (values[0] - values[1]) / step_penalty
