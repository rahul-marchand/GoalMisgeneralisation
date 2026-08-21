"""Reading the route model's residual stream in the shape the per-cell probes expect.

The DRC probe reads a ``(height, width, channels)`` grid captured at t=0 and
asks a linear readout, applied identically at every cell, whether that cell is
on the route the agent then walks. The route model's analogue is its residual
stream at the maze-token positions - one vector per cell - taken before any
action token exists. Because attention over the prefix is bidirectional and
the prefix never sees an action (``model.prefix_mask``), that grid is a
function of the maze alone, which is what makes the analogy exact.

:func:`capture` returns objects with the attributes
:mod:`goalmisgen.analysis.probes` reads - ``features``, ``observation``,
``visited``, ``visit_step``, ``distance`` - so the existing ``probe``,
``probe_by_distance`` and ``auc_interval`` run unchanged. The labels come from
a replay of the model's own greedy route by default; :func:`relabel` swaps in
the route to a named objective, which is how "does the probe read the optimal
route or the colour-0 route" is asked.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.envs.level import Level
from goalmisgen.envs.solver import distance_field, path_to_objective
from goalmisgen.offline.decode import Decoded, greedy_decode, replay_all
from goalmisgen.offline.demos import NO_ACTION, DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


@dataclasses.dataclass
class CellRollout:
    """One level: the per-cell residual before the first action, and the route."""

    features: np.ndarray
    """(size, size, d_model) - residual stream at the maze tokens, one layer."""

    observation: np.ndarray
    """(size, size, channels) - the model's input, for the baseline probe."""

    visited: np.ndarray
    visit_step: np.ndarray
    distance: np.ndarray
    info: dict
    level: Level


@functools.lru_cache(maxsize=8)
def _residuals_fn(config: ModelConfig):
    model = RoutePrefixLM(config)

    @jax.jit
    def residuals(params, observations):
        actions = jnp.full((observations.shape[0], config.max_actions), NO_ACTION, dtype=jnp.int32)
        _, streams = model.apply(params, observations, actions)
        return jnp.stack([s[:, : config.n_cells] for s in streams], axis=0)

    return residuals


def cell_residuals(model: RoutePrefixLM, params, observations: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """``(n_layers + 1, B, size, size, d_model)``: embedding, then after each block.

    Captured with no action tokens present; the prefix cannot see them anyway,
    so this is the same grid the model decodes from.
    """
    cfg = model.config
    fn = _residuals_fn(cfg)
    chunks = []
    for start in range(0, len(observations), batch_size):
        out = np.asarray(fn(params, jnp.asarray(observations[start : start + batch_size])))
        chunks.append(out.reshape(out.shape[0], out.shape[1], cfg.size, cfg.size, cfg.d_model))
    return np.concatenate(chunks, axis=1)


def capture(
    model: RoutePrefixLM,
    params,
    demos: DemoSet,
    indices: np.ndarray,
    layer: int | None,
    reader_params=None,
    decoded: Decoded | None = None,
) -> list[CellRollout]:
    """Rollouts labelled by the model's own greedy route, featured by ``layer``.

    ``layer`` is a depth (0 = the embedding, ``n`` = after block ``n``) or
    ``None`` for every depth concatenated - the closest match to the DRC probe,
    which reads all three recurrent layers at once.

    ``reader_params`` reads the residuals out of a different network - an
    untrained one of the same shape is the control - while ``params`` still
    decides the routes, so the labels are held fixed across arms. ``decoded``
    lets a caller reuse routes it has already computed.
    """
    indices = np.asarray(indices)
    observations = demos.observations(indices)
    if decoded is None:
        decoded = greedy_decode(model, params, observations)
    outcomes = replay_all(demos, indices, decoded)
    streams = cell_residuals(model, params if reader_params is None else reader_params, observations)
    streams = np.concatenate(list(streams), axis=-1) if layer is None else streams[layer]

    rollouts = []
    for row, index in enumerate(indices):
        level = demos.level(int(index))
        rollouts.append(
            CellRollout(
                features=streams[row],
                observation=observations[row],
                visited=_pad(outcomes[row]["visited"], demos.size),
                visit_step=_pad(outcomes[row]["visit_step"], demos.size, fill=-1),
                distance=_pad(distance_field(level.walls, level.agent_start), demos.size, fill=-1),
                info=outcomes[row],
                level=level,
            )
        )
    return rollouts


def _pad(grid: np.ndarray, size: int, fill=False) -> np.ndarray:
    """Embed a true-size grid in the padded observation frame."""
    if grid.shape == (size, size):
        return grid
    out = np.full((size, size), fill, dtype=grid.dtype)
    out[: grid.shape[0], : grid.shape[1]] = grid
    return out


def relabel(rollouts: list[CellRollout], objective: str) -> list[CellRollout]:
    """The same features, labelled by the route to a named objective.

    ``objective`` is ``"optimal"`` (the expert's target) or ``"feature0"`` (the
    objective carrying colour 0). Where the two coincide - every level at
    rho=1.0 - the relabelled rollouts are the expert's. Where they differ, the
    probe fitted on the model's own routes at rho=1.0 can be scored against
    each, and which one it reads is the question.
    """
    out = []
    for rollout in rollouts:
        level = rollout.level
        if objective == "optimal":
            index = int(rollout.info["optimal_index"])
        elif objective == "feature0":
            index = next(k for k, o in enumerate(level.objectives) if o.feature_id == 0)
        else:
            raise ValueError(f"unknown objective {objective!r}; expected 'optimal' or 'feature0'")
        path = path_to_objective(level, index)
        if path is None:
            continue
        size = rollout.visited.shape[0]
        visited = np.zeros((size, size), dtype=bool)
        visit_step = np.full((size, size), -1, dtype=np.int16)
        for step, cell in enumerate(path):
            visited[cell] = True
            visit_step[cell] = step
        out.append(dataclasses.replace(rollout, visited=visited, visit_step=visit_step))
    return out
