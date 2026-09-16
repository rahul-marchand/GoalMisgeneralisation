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
from goalmisgen.offline.demos import NO_ACTION, level_info, outcome_info

HEALTHY = 9
"""The engine's full health, food, drink and energy."""


def _cpu():
    """The device the engine runs on: always the CPU.

    ``craftax_step`` is thousands of tiny ops, and a 64-step scan of it over a
    thousand worlds is launch-bound on a GPU: 400 s per evaluation set on an
    L4 shared with training, against 2.5 s on one CPU core. The model decodes
    on the accelerator; the world it is scored in does not need one.
    """
    return jax.devices("cpu")[0]


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
    """An empty grass world with a healthy, empty-handed player at the origin.

    Built from **host** arrays, not device arrays: a batch is assembled with
    ``np.stack`` and moved to the device once per field by :func:`stack_states`.
    Stacking a thousand device scalars with ``jnp.stack`` compiles a
    thousand-operand concatenate per field - minutes per evaluation set, on the
    CPU, at every checkpoint - which is how the first pod run stalled at step 0.
    """
    static = static_params(size)

    def mobs(count: int) -> Mobs:
        return Mobs(
            position=np.zeros((count, 2), dtype=np.int32),
            health=np.zeros(count, dtype=np.int32),
            mask=np.zeros(count, dtype=np.bool_),
            attack_cooldown=np.zeros(count, dtype=np.int32),
        )

    return EnvState(
        map=np.full((size, size), int(Block.GRASS), dtype=np.int32),
        mob_map=np.zeros((size, size), dtype=np.bool_),
        player_position=np.zeros(2, dtype=np.int32),
        player_direction=np.int32(int(Action.DOWN)),
        player_health=np.int32(HEALTHY),
        player_food=np.int32(HEALTHY),
        player_drink=np.int32(HEALTHY),
        player_energy=np.int32(HEALTHY),
        is_sleeping=np.bool_(False),
        player_recover=np.float32(0.0),
        player_hunger=np.float32(0.0),
        player_thirst=np.float32(0.0),
        player_fatigue=np.float32(0.0),
        inventory=Inventory(*(np.int32(0) for _ in dataclasses.fields(Inventory))),
        zombies=mobs(static.max_zombies),
        cows=mobs(static.max_cows),
        skeletons=mobs(static.max_skeletons),
        arrows=mobs(static.max_arrows),
        arrow_directions=np.zeros((static.max_arrows, 2), dtype=np.int32),
        growing_plants_positions=np.zeros((static.max_growing_plants, 2), dtype=np.int32),
        growing_plants_age=np.zeros(static.max_growing_plants, dtype=np.int32),
        growing_plants_mask=np.zeros(static.max_growing_plants, dtype=np.bool_),
        light_level=np.float32(1.0),
        achievements=np.zeros(len(engine_constants.Achievement), dtype=np.bool_),
        state_rng=np.zeros(2, dtype=np.uint32),  # never read: craftax_step draws from the key it is given
        timestep=np.int32(0),
    )


def state_from_tiles(tiles: np.ndarray, agent: tuple[int, int], direction: int, tools: Sequence[str] = ()) -> EnvState:
    """An engine state holding exactly these tiles, with the player placed, facing, and equipped."""
    tiles = np.asarray(tiles)
    size = tiles.shape[0]
    if tiles.shape != (size, size):
        raise ValueError(f"worlds are square, got {tiles.shape}")
    blank = blank_state(size)
    return blank.replace(
        map=np.asarray(tiles, dtype=np.int32),
        player_position=np.asarray(agent, dtype=np.int32),
        player_direction=np.int32(direction),
        inventory=blank.inventory.replace(**{tool: np.int32(1) for tool in tools}),
    )


def build_state(level: Level, task: CraftaxTask) -> EnvState:
    """The level as an engine state: its tiles, the player placed and facing, the tools in hand."""
    return state_from_tiles(task.tiles(level), level.agent_start, task.start_direction, task.tools)


def stack_states(states: Sequence[EnvState]) -> EnvState:
    """A batch of states as one pytree with a leading batch axis, on the engine's device."""
    cpu = _cpu()
    stacked = jax.tree_util.tree_map(lambda *leaves: np.stack([np.asarray(leaf) for leaf in leaves]), *states)
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu), stacked)


def render(state: EnvState, feature_values: Sequence[float], task, hide_values: bool = False) -> np.ndarray:
    """The model's observation drawn from a single live state, in the task's own layout.

    The task owns the layout (``task.observe``); this is the invariant that
    what the model trained on is a function of engine state, tested by holding
    it equal to the stored observation.
    """
    row, col = (int(v) for v in np.asarray(state.player_position))
    return task.observe(np.asarray(state.map), (row, col), feature_values, hide_values)


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
    cpu = _cpu()
    states = jax.tree_util.tree_map(lambda x: jax.device_put(np.asarray(x), cpu), states)
    size = int(states.map.shape[-1])
    batch, n_steps = actions.shape
    rollout = _rollout_fn(
        size, n_steps, step_limit, tuple(int(k) for k in task.kinds), tuple(int(k) for k in task.interactable)
    )
    with jax.default_device(cpu):
        keys = jax.random.split(jax.random.PRNGKey(seed), batch)
        final, steps, out = rollout(states, jax.device_put(actions, cpu), keys)
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
