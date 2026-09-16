"""Receding-horizon decoding: one action per forward pass, the engine in the loop.

The route model still emits a whole route from the state it is shown, but only
its first action is executed; the engine steps, the state is observed again,
and the model is asked again. A bump or a cut tree changes what the model
sees, so it recovers instead of drifting, which the open-loop prefix-LM of
stage 2 could not.

The world is stepped with :func:`goalmisgen.craftax.engine.step_batch`, whose
per-step keys match :func:`engine.run`'s, so the routes this produces replay
to the same trajectories and are scored by the ordinary replay - nothing
downstream of a :class:`~goalmisgen.offline.decode.Decoded` changes.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.craftax import engine, simulate
from goalmisgen.craftax.blocks import Action
from goalmisgen.offline.decode import Decoded
from goalmisgen.offline.demos import NO_ACTION
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


@functools.lru_cache(maxsize=8)
def _first_token_fn(config: ModelConfig):
    """The model's first output position, moves through EOS, jitted once per shape."""
    model = RoutePrefixLM(config)

    @jax.jit
    def logits(params, observations):
        actions = jnp.full((observations.shape[0], config.max_actions), NO_ACTION, dtype=jnp.int32)
        out, _ = model.apply(params, observations, actions)
        return out[:, 0, : config.n_classes]

    return logits


def rollout(
    model: RoutePrefixLM, params, demos, indices: np.ndarray, batch_size: int = 512, seed: int = 0, policy=None
) -> Decoded:
    """Greedy receding-horizon routes for ``indices`` of a receding crafting set.

    ``policy`` overrides the model: a callable ``(observations, t) -> actions``,
    used by the tests to drive the loop with the expert's own routes.
    """
    cfg = model.config
    task = demos.task
    step_limit = int(demos.meta["step_limit"])
    first = _first_token_fn(cfg)
    indices = np.asarray(indices)
    all_actions, all_lengths, all_eos = [], [], []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batch = len(chunk)
        fields = [demos.level(int(i)) for i in chunk]
        values = demos.feature_values(chunk)
        states = engine.stack_states([engine.build_state(f, task) for f in fields])
        keys = engine.initial_keys(batch, seed)
        actions = np.full((batch, cfg.max_actions), NO_ACTION, dtype=np.int32)
        lengths = np.full(batch, cfg.max_actions, dtype=np.int32)
        finished = np.zeros(batch, dtype=bool)
        eos = np.zeros(batch, dtype=bool)
        for t in range(cfg.max_actions):
            inventories = np.stack([np.asarray(getattr(states.inventory, f)) for f in simulate.INVENTORY], axis=-1)
            observations = task.observe_batch(
                np.asarray(states.map),
                np.asarray(states.player_position),
                values,
                demos.hide_values,
                np.asarray(states.player_direction),
                inventories,
            )
            if policy is None:
                token = np.asarray(jnp.argmax(first(params, jnp.asarray(observations)), axis=-1))
            else:
                token = np.asarray(policy(observations, t))
            stopping = (token == cfg.eos) & ~finished
            eos |= stopping
            lengths[stopping] = t
            finished |= stopping
            if finished.all():
                break
            act = np.where(finished, int(Action.NOOP), token).astype(np.int32)
            actions[~finished, t] = act[~finished]
            states, keys, collected = engine.step_batch(states, task, act, keys, step_limit)
            done_now = collected.any(axis=-1) & ~finished
            lengths[done_now] = t + 1
            finished |= done_now | (t + 1 >= step_limit)
        all_actions.append(actions)
        all_lengths.append(lengths)
        all_eos.append(eos)
    return Decoded(np.concatenate(all_actions), np.concatenate(all_lengths), np.concatenate(all_eos))
