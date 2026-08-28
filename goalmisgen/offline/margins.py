"""First-action logit margins over constructed presentation orbits.

The route model is deterministic, so its per-level deviation from the utility
rule is not noise but an unmeasured function of the presentation. This module
supplies the pieces for decomposing that function along designed orbits:

- **the colour swap** - the same maze with the two colour channels exchanged.
  With values hidden the swapped observation is indistinguishable from a
  natural level, and anything keyed to *which location carries which role*
  inverts under it while maze- or colour-keyed effects survive;
- **constructed agent starts** - the sampler places agents uniformly over free
  cells (``MazeLevelSampler``), so a start redrawn under the same rule stays
  in-distribution while every path length changes;
- **the margin** - the model's first-action preference between moves that
  strictly approach one objective and moves that strictly approach the other,
  read from the SEP logits at float precision. Moves that approach both carry
  no information about the choice and are excluded from both sides.

Margins are a *readout to be validated*, not a definition of choice: callers
must check them against decoded behaviour before leaning on them.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.envs.level import Level
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL
from goalmisgen.envs.solver import (
    MOVES,
    UNREACHABLE,
    distance_field,
    shortest_path,
    walls_blocking_other_objectives,
)
from goalmisgen.offline.demos import NO_ACTION
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


def swap_colours(observations: np.ndarray) -> np.ndarray:
    """The role swap: colour channels exchanged, everything else untouched."""
    out = observations.copy()
    c0, c1 = FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1
    out[..., [c0, c1]] = observations[..., [c1, c0]]
    return out


def move_agent(observations: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """The same levels with the agent placed at ``cells`` ((B, 2) row, col)."""
    out = observations.copy()
    out[..., AGENT_CHANNEL] = 0.0
    rows = np.arange(len(out))
    out[rows, cells[:, 0], cells[:, 1], AGENT_CHANNEL] = 1.0
    return out


def objective_fields(level: Level) -> np.ndarray:
    """(K, H, W) int - distance from every cell to each objective, routed
    around the others; ``UNREACHABLE`` where blocked. One BFS per objective
    serves every start of the maze."""
    return np.stack(
        [
            distance_field(walls_blocking_other_objectives(level, k), objective.position)
            for k, objective in enumerate(level.objectives)
        ]
    )


def approach_sets(fields: np.ndarray, cell: tuple[int, int]) -> tuple[tuple[int, ...], ...]:
    """Per objective, the moves from ``cell`` that strictly shorten its route
    and do not shorten the other's. Empty tuples mark a degenerate cell."""
    height, width = fields.shape[1:]
    toward = []
    for k in range(len(fields)):
        moves = []
        here = fields[k][cell]
        for action, (dr, dc) in enumerate(MOVES):
            r, c = cell[0] + dr, cell[1] + dc
            if 0 <= r < height and 0 <= c < width and here != UNREACHABLE:
                if fields[k][r, c] != UNREACHABLE and fields[k][r, c] == here - 1:
                    moves.append(action)
        toward.append(set(moves))
    return tuple(tuple(sorted(toward[k] - toward[1 - k])) for k in range(2))


def divergence_cell(level: Level, start: tuple[int, int]) -> tuple[int, int] | None:
    """The last cell the two shortest routes from ``start`` share - the fork.

    The margin read there is the commitment decision: distance error along the
    shared corridor cancels out of it by construction. Returns ``start`` itself
    when the routes part immediately, and ``None`` when either objective is
    unreachable. In looped layouts the fork inherits ``shortest_path``'s
    deterministic tie-break, so a tie can report an earlier fork than the
    behavioural decision region - a conservative artifact, not a wrong cell.
    """
    paths = []
    for k, objective in enumerate(level.objectives):
        path = shortest_path(walls_blocking_other_objectives(level, k), start, objective.position)
        if path is None:
            return None
        paths.append(path)
    fork = start
    for a, b in zip(paths[0], paths[1]):
        if a != b:
            break
        fork = a
    return fork


@functools.lru_cache(maxsize=8)
def _first_logits_fn(config: ModelConfig):
    model = RoutePrefixLM(config)

    @jax.jit
    def logits(params, observations):
        actions = jnp.full((observations.shape[0], config.max_actions), NO_ACTION, dtype=jnp.int32)
        out, _ = model.apply(params, observations, actions)
        return out[:, 0, : config.n_actions]

    return logits


def first_action_logits(model: RoutePrefixLM, params, observations: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """(B, n_actions) - the SEP position's move logits, the decision readout."""
    fn = _first_logits_fn(model.config)
    chunks = []
    for start in range(0, len(observations), batch_size):
        chunks.append(np.asarray(fn(params, jnp.asarray(observations[start : start + batch_size]))))
    return np.concatenate(chunks)


def margin(logits: np.ndarray, toward_first: tuple[int, ...], toward_second: tuple[int, ...]) -> float:
    """Preference for approaching the first objective over the second, in logits.

    ``max`` rather than logsumexp: the decode is greedy, so the best move on
    each side is what competes. NaN when either side has no exclusive move.
    """
    if not toward_first or not toward_second:
        return float("nan")
    return float(np.max(logits[list(toward_first)]) - np.max(logits[list(toward_second)]))
