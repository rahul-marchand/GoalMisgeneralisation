"""Building Craftax states from levels, and walking routes through the real engine.

The only module in the package that imports Craftax, and therefore JAX. It has
three jobs:

- :func:`build_state` turns a :class:`~goalmisgen.envs.level.Level` and a
  :class:`~goalmisgen.craftax.demos.CraftaxTask` into an ``EnvState`` by hand.
  The engine's own reset cannot be used: it generates fractal-noise terrain and
  refuses maps smaller than 16 cells, whereas ``craftax_step`` only needs the
  fields to have consistent shapes.
- :func:`render` draws the model's observation from a live state, in the maze's
  channel layout. The invariant *observation = f(engine state)* is exact under
  hidden values; with values shown, the values come from the level, because
  the engine holds no objective values.
- :func:`replay_batch` executes decoded routes with ``craftax_step`` inside a
  ``lax.scan`` and returns the outcome dict the maze replay returns, plus what
  only an engine can report: ``walls_mined`` (a wood pickaxe digs through
  stone), ``wasted_actions`` and the visited ``positions``.

Engine facts the code leans on, verified against ``game_logic.py``: ores are
solid; a move into a solid block does not move the player but turns them to
face it; ``do_action`` runs before ``move_player`` within a step, so DO uses the
facing set by the previous step; each ore checks only its own pickaxe flag;
DO facing grass rolls a sapling on the rng, so a fixed key makes the replay
deterministic; energy falls to 8 around step 31, after which SLEEP puts the
player to sleep and every action is NOOP until it recovers. None of these
touch an expert route, and all of them are exercised by the tests.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic import constants as engine_constants
from craftax.craftax_classic.envs.craftax_state import EnvParams, EnvState, Inventory, Mobs, StaticEnvParams
from craftax.craftax_classic.game_logic import craftax_step

from goalmisgen.craftax.blocks import INVENTORY_FIELD, Action, Block
from goalmisgen.craftax.demos import CraftaxTask
from goalmisgen.envs.level import Level
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL
from goalmisgen.offline.demos import NO_ACTION, level_info, outcome_info

HEALTHY = 9
"""The engine's full health, food, drink and energy."""


@functools.lru_cache(maxsize=None)
def static_params(size: int) -> StaticEnvParams:
    return StaticEnvParams(map_size=(size, size))


@functools.lru_cache(maxsize=None)
def env_params(step_limit: int) -> EnvParams:
    """No mobs ever spawn; the episode's own step limit is the engine's."""
    return EnvParams(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
        max_timesteps=step_limit,
    )


def blank_state(size: int) -> EnvState:
    """An empty grass world with a healthy, empty-handed player at the origin."""
    static = static_params(size)

    def mobs(count: int) -> Mobs:
        return Mobs(
            position=jnp.zeros((count, 2), dtype=jnp.int32),
            health=jnp.zeros(count, dtype=jnp.int32),
            mask=jnp.zeros(count, dtype=jnp.bool_),
            attack_cooldown=jnp.zeros(count, dtype=jnp.int32),
        )

    return EnvState(
        map=jnp.full((size, size), int(Block.GRASS), dtype=jnp.int32),
        mob_map=jnp.zeros((size, size), dtype=jnp.bool_),
        player_position=jnp.zeros(2, dtype=jnp.int32),
        player_direction=jnp.int32(int(Action.DOWN)),
        player_health=jnp.int32(HEALTHY),
        player_food=jnp.int32(HEALTHY),
        player_drink=jnp.int32(HEALTHY),
        player_energy=jnp.int32(HEALTHY),
        is_sleeping=jnp.bool_(False),
        player_recover=jnp.float32(0.0),
        player_hunger=jnp.float32(0.0),
        player_thirst=jnp.float32(0.0),
        player_fatigue=jnp.float32(0.0),
        inventory=Inventory(*(jnp.int32(0) for _ in dataclasses.fields(Inventory))),
        zombies=mobs(static.max_zombies),
        cows=mobs(static.max_cows),
        skeletons=mobs(static.max_skeletons),
        arrows=mobs(static.max_arrows),
        arrow_directions=jnp.zeros((static.max_arrows, 2), dtype=jnp.int32),
        growing_plants_positions=jnp.zeros((static.max_growing_plants, 2), dtype=jnp.int32),
        growing_plants_age=jnp.zeros(static.max_growing_plants, dtype=jnp.int32),
        growing_plants_mask=jnp.zeros(static.max_growing_plants, dtype=jnp.bool_),
        light_level=jnp.float32(1.0),
        achievements=jnp.zeros(len(engine_constants.Achievement), dtype=jnp.bool_),
        state_rng=jax.random.PRNGKey(0),
        timestep=jnp.int32(0),
    )


def state_from_tiles(tiles: np.ndarray, agent: tuple[int, int], direction: int, tools: Sequence[str] = ()) -> EnvState:
    """An engine state holding exactly these tiles, with the player placed, facing, and equipped."""
    tiles = np.asarray(tiles)
    size = tiles.shape[0]
    if tiles.shape != (size, size):
        raise ValueError(f"worlds are square, got {tiles.shape}")
    blank = blank_state(size)
    return blank.replace(
        map=jnp.asarray(tiles, dtype=jnp.int32),
        player_position=jnp.asarray(agent, dtype=jnp.int32),
        player_direction=jnp.int32(direction),
        inventory=blank.inventory.replace(**{tool: jnp.int32(1) for tool in tools}),
    )


def build_state(level: Level, task: CraftaxTask) -> EnvState:
    """The level as an engine state: its tiles, the player placed and facing, the tools in hand."""
    return state_from_tiles(task.tiles(level), level.agent_start, task.start_direction, task.tools)


def stack_states(states: Sequence[EnvState]) -> EnvState:
    """A batch of states as one pytree with a leading batch axis."""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *states)


def render(state: EnvState, feature_values: Sequence[float], task: CraftaxTask, hide_values: bool = False) -> np.ndarray:
    """``(size, size, channels)`` float32 in the maze's channel layout, from a single state.

    Wall = stone (mined stone is path, and walkable, so it drops out); agent
    one-hot; one channel per kind; and, unless hidden, the value of feature
    ``k`` (``feature_values[k]``) on the cells of kind ``k``.
    """
    tiles = np.asarray(state.map)
    size = tiles.shape[0]
    n_channels = FIRST_FEATURE_CHANNEL + task.n_features + (0 if hide_values else 1)
    observation = np.zeros((size, size, n_channels), dtype=np.float32)
    observation[..., WALL_CHANNEL] = tiles == int(Block.STONE)
    row, col = (int(v) for v in np.asarray(state.player_position))
    observation[row, col, AGENT_CHANNEL] = 1.0
    value_channel = FIRST_FEATURE_CHANNEL + task.n_features
    for k, kind in enumerate(task.kinds):
        mask = tiles == kind
        observation[mask, FIRST_FEATURE_CHANNEL + k] = 1.0
        if not hide_values:
            observation[mask, value_channel] = float(feature_values[k])
    return observation


# ----------------------------------------------------------------------
# Rollouts
# ----------------------------------------------------------------------

_MOVE_LOW, _MOVE_HIGH = int(Action.LEFT), int(Action.DOWN)


@functools.lru_cache(maxsize=16)
def _rollout_fn(size: int, n_steps: int, step_limit: int, kinds: tuple[int, ...], interactable: tuple[int, ...]):
    """One jitted scan per (world size, route length, step limit, kinds, interactable blocks).

    ``kinds`` are the objectives: the first inventory rise among them ends the
    route. ``interactable`` are the solid blocks a blocked move may legitimately
    turn to face (the kinds, plus trees and stone for a task that gathers).
    """
    static = static_params(size)
    params = env_params(step_limit)
    interactable_array = jnp.asarray(interactable, dtype=jnp.int32)
    fields = [INVENTORY_FIELD[Block(kind)] for kind in kinds]
    directions = jnp.asarray(engine_constants.DIRECTIONS)
    step = jax.vmap(lambda key, state, action: craftax_step(key, state, action, params, static)[0])

    def counts(states: EnvState) -> jnp.ndarray:
        return jnp.stack([getattr(states.inventory, field) for field in fields], axis=-1)

    def inventory_matrix(states: EnvState) -> jnp.ndarray:
        return jnp.stack([getattr(states.inventory, f.name) for f in dataclasses.fields(Inventory)], axis=-1)

    def per_env(mask: jnp.ndarray, new, old):
        return jnp.where(mask.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

    @jax.jit
    def rollout(states: EnvState, actions: jnp.ndarray, keys: jnp.ndarray):
        batch = actions.shape[0]
        rows = jnp.arange(batch)

        def body(carry, t):
            states, finished, steps, keys = carry
            action = actions[:, t]
            valid = (action >= 0) & ~finished
            action = jnp.where(valid, action, int(Action.NOOP))
            split = jax.vmap(jax.random.split)(keys)
            keys, subkeys = split[:, 0], split[:, 1]
            new = step(subkeys, states, action)

            collected = (counts(new) > counts(states)) & valid[:, None]
            is_move = (action >= _MOVE_LOW) & (action <= _MOVE_HIGH)
            moved = jnp.any(new.player_position != states.player_position, axis=-1)
            faced_position = jnp.clip(states.player_position + directions[action], 0, size - 1)
            faced = states.map[rows, faced_position[:, 0], faced_position[:, 1]]
            faced_interactable = jnp.isin(faced, interactable_array)
            asleep = states.is_sleeping  # the engine turns every action into NOOP
            illegal = valid & ~asleep & is_move & ~moved & ~faced_interactable
            walls_mined = valid & (new.inventory.stone > states.inventory.stone)
            # Wasted: an action that changed nothing the task can see - no move,
            # no turn, no inventory or map change - or anything done asleep.
            effect = (
                moved
                | (new.player_direction != states.player_direction)
                | jnp.any(inventory_matrix(new) != inventory_matrix(states), axis=-1)
                | jnp.any(new.map != states.map, axis=(-2, -1))
            )
            wasted = valid & (asleep | ~effect | (action == int(Action.SLEEP)))

            states = jax.tree_util.tree_map(functools.partial(per_env, valid), new, states)
            steps = steps + valid.astype(jnp.int32)
            finished = finished | collected.any(axis=-1) | (steps >= step_limit)
            out = dict(
                position=states.player_position,
                valid=valid,
                collected=collected,
                illegal=illegal,
                walls_mined=walls_mined,
                wasted=wasted,
            )
            return (states, finished, steps, keys), out

        init = (states, jnp.zeros(batch, dtype=jnp.bool_), jnp.zeros(batch, dtype=jnp.int32), keys)
        (states, _, steps, _), outputs = jax.lax.scan(body, init, jnp.arange(n_steps))
        return states, steps, outputs

    return rollout


@dataclasses.dataclass(frozen=True)
class Rollout:
    """What the engine reports for a batch of routes, arrays over ``(batch, step)``."""

    final: EnvState
    steps: np.ndarray  # (B,) actions executed
    position: np.ndarray  # (B, T, 2) after each step
    valid: np.ndarray  # (B, T)
    collected: np.ndarray  # (B, T, K)
    illegal: np.ndarray  # (B, T)
    walls_mined: np.ndarray  # (B, T)
    wasted: np.ndarray  # (B, T)


def run(states: EnvState, task, actions: np.ndarray, step_limit: int, seed: int = 0) -> Rollout:
    """Step a batch of states through ``actions`` ``(B, T)``; ``NO_ACTION`` ends a route."""
    actions = np.asarray(actions, dtype=np.int32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be (batch, steps), got {actions.shape}")
    size = int(states.map.shape[-1])
    batch, n_steps = actions.shape
    rollout = _rollout_fn(
        size, n_steps, step_limit, tuple(int(k) for k in task.kinds), tuple(int(k) for k in task.interactable)
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), batch)
    final, steps, out = rollout(states, jnp.asarray(actions), keys)
    swap = lambda x: np.asarray(jnp.swapaxes(x, 0, 1))  # noqa: E731 - (T, B, ...) -> (B, T, ...)
    return Rollout(
        final=final,
        steps=np.asarray(steps),
        position=swap(out["position"]),
        valid=swap(out["valid"]),
        collected=swap(out["collected"]),
        illegal=swap(out["illegal"]),
        walls_mined=swap(out["walls_mined"]),
        wasted=swap(out["wasted"]),
    )


def outcome(
    level: Level, task: CraftaxTask, rollout: Rollout, row: int, step_penalty: float, step_limit: int, emitted_eos: bool
) -> dict:
    """The outcome dict for one route of a rollout: the maze's keys and the engine's extras."""
    solution = task.solution(level, step_penalty, step_limit)
    info = level_info(level, solution)

    steps = int(rollout.steps[row])
    collected = rollout.collected[row, :steps]
    reached = None
    if collected.any():
        first = int(np.argmax(collected.any(axis=-1)))
        kind_index = int(np.argmax(collected[first]))
        (reached,) = [index for index, objective in enumerate(level.objectives) if objective.feature_id == kind_index]
    info.update(outcome_info(level, solution, reached, steps, step_penalty))

    positions = np.concatenate([np.asarray([level.agent_start], dtype=np.int32), rollout.position[row, :steps]])
    height, width = level.shape
    visited = np.zeros((height, width), dtype=bool)
    visit_step = np.full((height, width), -1, dtype=np.int16)
    for t, (r, c) in enumerate(positions):
        if not visited[r, c]:
            visited[r, c] = True
            visit_step[r, c] = t
    info.update(
        illegal_moves=int(rollout.illegal[row, :steps].sum()),
        emitted_eos=bool(emitted_eos),
        visited=visited,
        visit_step=visit_step,
        walls_mined=int(rollout.walls_mined[row, :steps].sum()),
        wasted_actions=int(rollout.wasted[row, :steps].sum()),
        positions=positions,
    )
    return info


def replay_batch(
    levels: Sequence[Level],
    task,
    actions: np.ndarray,
    step_penalty: float,
    step_limit: int,
    emitted_eos: Sequence[bool] | None = None,
    seed: int = 0,
    states: EnvState | None = None,
) -> list[dict]:
    """Execute one route per level through the engine and score each.

    ``states`` lets a caller start from states it has altered (a tool removed,
    a tile changed); by default they are built from the levels.
    """
    actions = np.asarray(actions, dtype=np.int32)
    if len(levels) != actions.shape[0]:
        raise ValueError(f"{len(levels)} levels but {actions.shape[0]} routes")
    if states is None:
        states = stack_states([build_state(level, task) for level in levels])
    if emitted_eos is None:
        emitted_eos = [True] * len(levels)
    rollout = run(states, task, actions, step_limit, seed)
    return [
        outcome(level, task, rollout, row, step_penalty, step_limit, bool(emitted_eos[row]))
        for row, level in enumerate(levels)
    ]


def replay(
    level: Level,
    task,
    actions: Sequence[int],
    step_penalty: float,
    step_limit: int,
    emitted_eos: bool = True,
    seed: int = 0,
) -> dict:
    """One route on one level; the batched form is the fast path."""
    actions = np.asarray(actions, dtype=np.int32)
    if actions.size == 0:
        actions = np.asarray([NO_ACTION], dtype=np.int32)
    return replay_batch([level], task, actions[None], step_penalty, step_limit, [emitted_eos], seed)[0]
