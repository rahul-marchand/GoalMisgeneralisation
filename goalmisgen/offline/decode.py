"""Turning an emitted route back into an episode the existing metrics can score.

A language model does not step an environment; it emits tokens. To compare it
with the DRC agents on the same footing, its greedy route is *replayed* under
the environment's rules - a move into a wall costs a step and goes nowhere,
reaching any objective ends the episode, the step limit truncates - and the
replay produces the same ``info`` dictionary ``MazeEnv`` would have, so
:mod:`goalmisgen.analysis.behaviour` summarises it unchanged. A test asserts
the replay and the environment agree step for step.

Two things an emitted route can do that an agent cannot: stop (EOS) before
reaching anything, and walk into walls. Both are reported.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.behaviour import BehaviourSummary, indifference_point, summarise, value_distance_decisions
from goalmisgen.envs.level import Level
from goalmisgen.envs.solver import MOVES, UNREACHABLE, solve
from goalmisgen.offline.demos import NO_ACTION, DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


@dataclasses.dataclass(frozen=True)
class Decoded:
    """Greedy routes for a batch: moves, their lengths, and whether EOS came."""

    actions: np.ndarray  # (B, max_actions) int32, NO_ACTION past the end
    lengths: np.ndarray  # (B,) int - moves before EOS, or max_actions if none
    emitted_eos: np.ndarray  # (B,) bool


DECODE_HEAD_BATCHES = 4096
"""Cap on ``batch x n_heads`` for one decode chunk. See :func:`decode_batch_size`."""


def decode_batch_size(model: RoutePrefixLM) -> int:
    """How many routes to decode at once, for a model of this width.

    A fixed chunk is the wrong unit. Decoding materialises the attention matrix,
    ``batch x heads x length x length``, and that is the whole memory bill -- so
    a chunk that is comfortable for a four-head model is four times too large
    for a sixteen-head one. Holding ``batch x heads`` constant makes the cost
    independent of the model's width instead.

    This is not hypothetical tidying. A fixed 1024 ran the 12M and 50M cells of
    the scaling grid out of memory on a 24 GB card *during evaluation*, after
    training itself had fitted comfortably: the probe that sized the campaign
    measured training and never measured a decode.
    """
    return max(32, DECODE_HEAD_BATCHES // model.config.n_heads)


def greedy_decode(model: RoutePrefixLM, params, observations: np.ndarray, batch_size: int | None = None) -> Decoded:
    """Argmax one token at a time until every route has ended.

    Recomputes the full sequence at every step rather than caching keys and
    values: the sequences are under two hundred tokens and the model is tiny,
    so the cache would be more code than the time it saves.

    ``batch_size`` is only how the work is chunked, never how much of it there
    is: every observation passed in is decoded, and the result does not depend
    on it. It defaults to :func:`decode_batch_size`.
    """
    cfg = model.config
    batch_size = decode_batch_size(model) if batch_size is None else batch_size
    next_token = _next_token(cfg)

    out_actions, out_lengths, out_eos = [], [], []
    for start in range(0, len(observations), batch_size):
        obs = jnp.asarray(observations[start : start + batch_size])
        batch = obs.shape[0]
        actions = np.full((batch, cfg.max_actions), NO_ACTION, dtype=np.int32)
        lengths = np.full(batch, cfg.max_actions, dtype=np.int32)
        finished = np.zeros(batch, dtype=bool)

        for position in range(cfg.max_actions + 1):
            token = np.asarray(next_token(params, obs, jnp.asarray(actions), position))
            ended = (token == cfg.eos) & ~finished
            lengths[ended] = position
            finished |= ended
            if finished.all():
                break
            if position < cfg.max_actions:
                actions[:, position] = np.where(finished, NO_ACTION, token)

        out_actions.append(actions)
        out_lengths.append(lengths)
        out_eos.append(finished)

    return Decoded(np.concatenate(out_actions), np.concatenate(out_lengths), np.concatenate(out_eos))


@functools.lru_cache(maxsize=8)
def _next_token(config: ModelConfig):
    """One jitted decode step per model shape.

    Built inside :func:`greedy_decode` it was recompiled on every call, and an
    evaluation at every checkpoint paid the compile each time.
    """
    model = RoutePrefixLM(config)

    @jax.jit
    def next_token(params, observations, actions, position):
        logits, _ = model.apply(params, observations, actions)
        return jnp.argmax(logits[:, position], axis=-1)

    return next_token


def replay(level: Level, actions: Sequence[int], step_penalty: float, step_limit: int, emitted_eos: bool = True) -> dict:
    """Walk ``actions`` on ``level`` under ``MazeEnv``'s rules; return its info.

    Returns the union of the environment's level info and outcome info, plus
    ``illegal_moves`` (moves into walls), ``emitted_eos``, and the ``visited``
    / ``visit_step`` grids the plan probe labels are built from.
    """
    solution = solve(level, step_penalty, step_limit=step_limit)
    height, width = level.shape
    visited = np.zeros((height, width), dtype=bool)
    visit_step = np.full((height, width), -1, dtype=np.int16)

    position = level.agent_start
    visited[position] = True
    visit_step[position] = 0
    goal = {objective.position: index for index, objective in enumerate(level.objectives)}

    reached = None
    steps = 0
    illegal = 0
    for action in actions:
        if action == NO_ACTION:
            break
        d_row, d_col = MOVES[int(action)]
        candidate = (position[0] + d_row, position[1] + d_col)
        inside = 0 <= candidate[0] < height and 0 <= candidate[1] < width
        if inside and not level.is_wall(candidate):
            position = candidate
        else:
            illegal += 1
        steps += 1
        if not visited[position]:
            visited[position] = True
            visit_step[position] = steps
        reached = goal.get(position)
        if reached is not None or steps >= step_limit:
            break

    optimal = solution.optimal_index
    info: dict = {
        "optimal_index": optimal,
        "optimal_feature_id": level.objectives[optimal].feature_id,
        "optimal_value": level.objectives[optimal].value,
        "optimal_distance": solution.distances[optimal],
        "utility_margin": solution.utility_margin,
        "is_ambiguous": solution.is_ambiguous,
        "level_size": height,
    }
    for index, objective in enumerate(level.objectives):
        distance = solution.distances[index]
        info[f"feature_{objective.feature_id}_value"] = objective.value
        info[f"feature_{objective.feature_id}_distance"] = UNREACHABLE if distance is None else distance

    walked = -step_penalty * steps
    if reached is None:
        info.update(reached_objective=False, episode_steps=steps, episode_return=walked)
    else:
        info.update(
            reached_objective=True,
            reached_index=reached,
            reached_feature_id=level.objectives[reached].feature_id,
            reached_value=level.objectives[reached].value,
            chose_optimal=reached in solution.optimal_indices,
            episode_steps=steps,
            episode_return=walked + level.objectives[reached].value,
        )
    info.update(illegal_moves=illegal, emitted_eos=bool(emitted_eos), visited=visited, visit_step=visit_step)
    return info


def replay_all(demos: DemoSet, indices: np.ndarray, decoded: Decoded) -> list[dict]:
    step_penalty = float(demos.meta["step_penalty"])
    step_limit = int(demos.meta["step_limit"])
    return [
        replay(demos.level(int(index)), decoded.actions[row], step_penalty, step_limit, bool(decoded.emitted_eos[row]))
        for row, index in enumerate(indices)
    ]


@dataclasses.dataclass(frozen=True)
class RouteSummary:
    """What the decoded routes did, in the environment's terms plus the model's."""

    behaviour: BehaviourSummary
    legal: float
    """Fraction of routes with no move into a wall."""

    emitted_eos: float
    """Fraction that ended the route themselves rather than hitting the cap."""

    matched_expert: float
    """Fraction whose route is the demonstration's route, move for move."""

    indifference: float
    """Distance gap at which the richer objective is taken half the time.

    The expert's is ``(value gap) / step_penalty``: 10 steps at (1.0, 0.5) and
    0.05. Comparable with the DRC exchange rates in ``results/``.
    """

    def as_row(self) -> dict[str, float]:
        b = self.behaviour
        return {
            "episodes": b.episodes,
            "reached": b.reached_objective,
            "chose_optimal": b.chose_optimal,
            "followed_feature_zero": b.followed_feature_zero,
            "ambiguous": b.ambiguous,
            "mean_return": b.mean_return,
            "mean_steps": b.mean_steps,
            "legal": self.legal,
            "emitted_eos": self.emitted_eos,
            "matched_expert": self.matched_expert,
            "indifference": self.indifference,
        }

    def __str__(self) -> str:
        b = self.behaviour
        return (
            f"reached {b.reached_objective:.1%}  optimal {b.chose_optimal:.1%}  "
            f"followed f0 {b.followed_feature_zero:.1%}  legal {self.legal:.1%}  "
            f"matched expert {self.matched_expert:.1%}  return {b.mean_return:.3f}  "
            f"steps {b.mean_steps:.1f}  indifference {self.indifference:.1f}"
        )


def summarise_routes(
    demos: DemoSet, indices: np.ndarray, decoded: Decoded, outcomes: list[dict], indifference: bool = True
) -> RouteSummary:
    """``indifference=False`` skips the exchange-rate fit and reports NaN for it.

    That fit is :func:`~goalmisgen.analysis.behaviour.indifference_point`, four
    thousand gradient steps, and it dominates everything else here by an order
    of magnitude: 39.5s against 6.6s of decoding and 2.7s of replay per 10,000
    levels. A caller that only wants the routes -- every sweep script does --
    should not pay for it, the more so since the fitted crossing is biased by
    saturation and the analyses read theirs off binned rates instead.
    """
    expert = demos.routes(indices)
    matched = np.all(decoded.actions == expert, axis=1)
    gaps, took_richer, _ = value_distance_decisions(outcomes) if indifference else (np.empty(0), np.empty(0), None)
    return RouteSummary(
        behaviour=summarise(outcomes),
        legal=float(np.mean([o["illegal_moves"] == 0 for o in outcomes])),
        emitted_eos=float(decoded.emitted_eos.mean()),
        matched_expert=float(matched.mean()),
        indifference=indifference_point(gaps, took_richer) if len(gaps) else float("nan"),
    )


def evaluate(
    model: RoutePrefixLM, params, demos: DemoSet, indices: np.ndarray, decoder=None, indifference: bool = True
) -> tuple[RouteSummary, Decoded, list[dict]]:
    """Decode, replay and summarise one held-out set.

    ``decoder`` defaults to :func:`greedy_decode`. Pass
    ``fast_decode.greedy_decode_cached`` for the same routes an order of
    magnitude faster; it is imported lazily by the caller rather than here, so
    this module keeps no dependency on it.
    """
    decoded = (decoder or greedy_decode)(model, params, demos.observations(indices))
    outcomes = replay_all(demos, indices, decoded)
    return summarise_routes(demos, indices, decoded, outcomes, indifference), decoded, outcomes


def build_model(config: ModelConfig) -> RoutePrefixLM:
    return RoutePrefixLM(config)
