"""The Craftax task against the real engine.

Invariants (Craftax.md): the expert's route, executed by ``craftax_step``,
collects the target in exactly ``distance`` steps and nothing else; the
rendered observation equals what the model trained on; the outcome dict is a
superset of the maze's with the shared keys agreeing; and the engine facts the
planner relies on (turn-on-blocked-move, DO-before-move, per-ore pickaxes,
diggable stone, sleep) behave as documented.
"""

from __future__ import annotations

import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from goalmisgen.craftax import engine
from goalmisgen.craftax.blocks import BEDROCK, Action, Block
from goalmisgen.craftax.demos import MOVE_TO_ACTION, CraftaxDemoSet, CraftaxTask
from goalmisgen.craftax.levels import OreFieldGenerator, generate_ore_fields
from goalmisgen.envs.level import Level, Objective
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.demonstrations import PROTOCOL_ATTRIBUTES, Demonstrations, load_demonstrations
from goalmisgen.offline.demos import NO_ACTION, DemoSet
from goalmisgen.offline.demos import replay as maze_replay

N_LEVELS = 120


@pytest.fixture(scope="module")
def dataset():
    sampler = MazeLevelSampler(generator=OreFieldGenerator(0.3), size_range=(15, 15))
    return generate_ore_fields(sampler, n_levels=N_LEVELS, seed=0, block_size=60)


@pytest.fixture(scope="module")
def demos(dataset) -> CraftaxDemoSet:
    return CraftaxDemoSet.generate(dataset, np.arange(N_LEVELS), rho=1.0, step_limit=120, max_actions=64)


def room(size: int = 7) -> np.ndarray:
    walls = np.ones((size, size), dtype=bool)
    walls[1:-1, 1:-1] = False
    return walls


# ----------------------------------------------------------------------
# The demonstration set
# ----------------------------------------------------------------------


def test_craftax_demos_carry_every_protocol_attribute(demos):
    missing = [name for name in PROTOCOL_ATTRIBUTES if not hasattr(demos, name)]
    assert not missing
    assert isinstance(demos, Demonstrations)
    assert demos.n_actions == 17
    assert demos.move_actions == MOVE_TO_ACTION
    assert demos.max_actions == 64 and demos.meta["step_limit"] == 120 and demos.inner.meta["step_limit"] == 119


def test_routes_are_canonical_shortest_paths_plus_do_minus_a_redundant_turn(demos):
    """Same length as the maze's path (plus DO, minus the turn after a straight run), canonically tie-broken."""
    from goalmisgen.craftax.routes import ORDER, distances_to
    from goalmisgen.envs.solver import MOVES, walls_blocking_other_objectives

    inner = demos.inner
    straight = 0
    for i in range(40):
        route = demos.routes([i])[0]
        target = int(demos.target[i])
        n = int(demos.lengths[i])
        assert route[n - 1] == Action.DO and (route[n:] == NO_ACTION).all()
        d = int(inner.distances[i, target])
        assert n in (d, d + 1), "engine cost is the maze distance plus DO, minus the turn after a straight run"
        straight += n == d
        # every move is the first, in ORDER, that shortens the distance to the target
        level = demos.level(i)
        field = distances_to(walls_blocking_other_objectives(level, target), level.objectives[target].position)
        here = level.agent_start
        for action in route[: n - 1]:
            move = MOVE_TO_ACTION.index(int(action))
            wanted = int(field[here]) - 1
            first = next(m for m in ORDER if field[here[0] + MOVES[m][0], here[1] + MOVES[m][1]] == wanted)
            assert move == first
            here = (here[0] + MOVES[move][0], here[1] + MOVES[move][1])
        assert int(field[here]) <= 1, "the route ends beside (or, after a dropped turn, still beside) the ore"
    assert 0 < straight < 40, "both cases occur"
    for i in range(40):
        for k in range(2):
            d = int(inner.distances[i, k])
            assert d <= int(demos.distances[i, k]) <= d + 1


def test_the_expert_chooses_by_engine_cost(demos):
    for i in range(40):
        level = demos.level(i)
        solution = demos.task.solution(level, 0.05, 120)
        assert solution.optimal_index == demos.target[i]
        assert tuple(int(c) for c in demos.distances[i]) == solution.distances


def test_observations_are_the_maze_observations(demos):
    assert np.array_equal(demos.observations([0, 1]), demos.inner.observations([0, 1]))
    hidden = demos.with_hidden_values()
    assert hidden.n_channels == demos.n_channels - 1


def test_save_load_round_trip_is_a_craftax_set(demos, tmp_path):
    demos.save(tmp_path / "demos")
    loaded = load_demonstrations(tmp_path / "demos", hide_values=True)
    assert isinstance(loaded, CraftaxDemoSet)
    assert loaded.task == demos.task and loaded.hide_values
    assert np.array_equal(loaded.routes([3]), demos.routes([3]))


def test_generating_demonstrations_never_imports_jax():
    probe = (
        "import sys; from goalmisgen.craftax import demos, levels; "
        "assert 'jax' not in sys.modules and 'craftax' not in sys.modules, "
        "[m for m in sys.modules if m in ('jax', 'craftax')]"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_task_rejects_trees_and_duplicate_kinds():
    with pytest.raises(ValueError, match="distinct"):
        CraftaxTask(kinds=(int(Block.COAL), int(Block.COAL)))
    with pytest.raises(ValueError, match="TREE"):
        CraftaxTask(kinds=(int(Block.TREE), int(Block.COAL)))
    assert CraftaxTask().tools == ("iron_pickaxe", "wood_pickaxe")


# ----------------------------------------------------------------------
# Planner against engine
# ----------------------------------------------------------------------


def test_expert_routes_collect_their_target_in_exactly_distance_steps(demos):
    indices = np.arange(N_LEVELS)
    outcomes = demos.replay_many(indices, demos.routes(indices), [True] * N_LEVELS)
    for i, info in enumerate(outcomes):
        target = int(demos.target[i])
        assert info["reached_objective"], i
        assert info["reached_index"] == target, i
        assert info["episode_steps"] == demos.distances[i, target] == demos.lengths[i], i
        assert info["illegal_moves"] == 0 and info["wasted_actions"] == 0 and info["walls_mined"] == 0, i
        assert info["chose_optimal"] or info["is_ambiguous"], i
    final = outcomes[0]
    assert final["positions"].shape == (final["episode_steps"] + 1, 2)


def test_the_other_ore_is_left_standing(demos):
    level = demos.level(0)
    states = engine.stack_states([engine.build_state(level, demos.task)])
    rollout = engine.run(states, demos.task, demos.routes([0]), step_limit=120)
    tiles = np.asarray(rollout.final.map[0])
    target, other = int(demos.target[0]), 1 - int(demos.target[0])
    assert tiles[level.objectives[target].position] == Block.PATH
    assert tiles[level.objectives[other].position] == demos.task.kinds[level.objectives[other].feature_id]


def test_batch_and_single_replay_agree(demos):
    indices = np.arange(6)
    batched = demos.replay_many(indices, demos.routes(indices), [True] * 6)
    for i in indices:
        single = demos.replay(int(i), demos.routes([i])[0])
        for key, value in single.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(value, batched[i][key]), key
            else:
                assert value == batched[i][key], key


# ----------------------------------------------------------------------
# Observation faithfulness
# ----------------------------------------------------------------------


def test_rendered_state_equals_the_training_observation(demos):
    for i in range(5):
        level = demos.level(i)
        state = engine.build_state(level, demos.task)
        values = [0.0] * demos.task.n_features
        for objective in level.objectives:
            values[objective.feature_id] = objective.value
        assert np.array_equal(engine.render(state, values, demos.task), demos.observations([i])[0])
        hidden = demos.with_hidden_values()
        assert np.array_equal(engine.render(state, values, demos.task, hide_values=True), hidden.observations([i])[0])


def test_rendering_after_the_route_shows_the_ore_mined_and_the_player_moved(demos):
    level = demos.level(0)
    states = engine.stack_states([engine.build_state(level, demos.task)])
    rollout = engine.run(states, demos.task, demos.routes([0]), step_limit=120)
    after = engine.render(jax_index(rollout.final, 0), [1.0, 0.5], demos.task)
    target = level.objectives[int(demos.target[0])]
    assert after[target.position][2 + target.feature_id] == 0.0, "mined ore is gone from its kind channel"
    assert after[target.position][0] == 0.0, "and it is path, not wall"
    assert after[..., 1].sum() == 1.0 and after[level.agent_start][1] == 0.0


def jax_index(states, row: int):
    import jax

    return jax.tree_util.tree_map(lambda x: x[row], states)


# ----------------------------------------------------------------------
# Outcome contract with the maze
# ----------------------------------------------------------------------


def test_outcome_keys_are_a_superset_of_the_mazes_and_shared_keys_agree(demos):
    inner = demos.inner
    for i in range(5):
        level = demos.level(i)
        maze = maze_replay(level, inner.routes([i])[0], 0.05, 119)
        craft = demos.replay(i, demos.routes([i])[0])
        assert set(maze) <= set(craft)
        if maze["optimal_index"] != craft["optimal_index"]:
            continue  # a near-tie the engine costs resolve differently; nothing else is comparable then
        for key in maze:
            if key.endswith("_distance") and maze[key] >= 0:
                assert craft[key] in (maze[key], maze[key] + 1), key
            elif key in ("episode_steps", "optimal_distance"):
                assert craft[key] in (maze[key], maze[key] + 1)
            elif key in ("episode_return", "utility_margin"):
                assert abs(craft[key] - maze[key]) <= 0.05 + 1e-9
            elif key in ("visited", "visit_step", "is_ambiguous", "chose_optimal"):
                continue  # the maze steps onto the objective; the engine stops beside it; ties may differ
            else:
                assert craft[key] == maze[key], key


# ----------------------------------------------------------------------
# Engine facts
# ----------------------------------------------------------------------


def corridor_level() -> Level:
    """Agent at (3,1) facing down; coal at (3,5) to the right, diamond at (1,3) above."""
    walls = room(7)
    return Level(walls=walls, agent_start=(3, 1), objectives=(Objective((3, 5), 1.0, 0), Objective((1, 3), 0.5, 1)))


def test_a_blocked_move_turns_the_player_and_do_then_mines():
    task = CraftaxTask()
    level = corridor_level()
    # RIGHT x3 -> (3,4) adjacent; RIGHT again is blocked by the coal (turns); DO mines.
    route = np.asarray([[Action.RIGHT] * 4 + [Action.DO] + [NO_ACTION] * 3])
    (info,) = engine.replay_batch([level], task, route, 0.05, 120)
    assert info["reached_objective"] and info["reached_feature_id"] == 0
    assert info["episode_steps"] == 5 and info["illegal_moves"] == 0, "the turn is not an illegal move"
    # Arriving beside the coal from above leaves the player facing DOWN, so DO
    # faces grass: nothing is collected and the DO is wasted. One more RIGHT
    # (blocked, a turn) and DO then mines.
    approach = [Action.UP, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.DOWN]
    route = np.asarray([approach + [Action.DO] + [NO_ACTION] * 2])
    (info,) = engine.replay_batch([level], task, route, 0.05, 120)
    assert not info["reached_objective"] and info["wasted_actions"] == 1 and info["episode_steps"] == 6
    route = np.asarray([approach + [Action.DO, Action.RIGHT, Action.DO]])
    (info,) = engine.replay_batch([level], task, route, 0.05, 120)
    assert info["reached_objective"] and info["episode_steps"] == 8 and info["illegal_moves"] == 0


def test_each_ore_checks_only_its_own_pickaxe():
    task = CraftaxTask()
    level = corridor_level()
    route = np.asarray([[Action.RIGHT] * 4 + [Action.DO] + [NO_ACTION] * 3])
    state = engine.build_state(level, task)
    no_wood = state.replace(inventory=state.inventory.replace(wood_pickaxe=jnp.int32(0)))
    (info,) = engine.replay_batch([level], task, route, 0.05, 120, states=engine.stack_states([no_wood]))
    assert not info["reached_objective"], "an iron pickaxe alone does not mine coal"


def test_walls_are_bedrock_and_cannot_be_dug():
    task = CraftaxTask()
    level = corridor_level()
    # LEFT into the border turns the player; DO on bedrock does nothing; LEFT again is still blocked.
    route = np.asarray([[Action.LEFT, Action.DO, Action.LEFT] + [NO_ACTION] * 5])
    (info,) = engine.replay_batch([level], task, route, 0.05, 120)
    assert info["walls_mined"] == 0 and info["wasted_actions"] == 2, "the DO had no effect, nor the repeated bump"
    assert info["illegal_moves"] == 2 and tuple(info["positions"][-1]) == (3, 1)
    states = engine.stack_states([engine.build_state(level, task)])
    tiles = np.asarray(states.map[0])
    assert tiles[3, 0] == BEDROCK and not (tiles == Block.STONE).any()


def test_moves_into_walls_are_illegal_and_the_step_limit_truncates():
    task = CraftaxTask()
    level = corridor_level()
    route = np.asarray([[Action.UP] * 10])  # (2,1), (1,1), then 8 blocked by the border
    (info,) = engine.replay_batch([level], task, route, 0.05, step_limit=6)
    assert info["episode_steps"] == 6 and info["illegal_moves"] == 4 and not info["reached_objective"]


def test_sleep_late_in_an_episode_freezes_the_player_and_is_reported_as_wasted():
    task = CraftaxTask()
    level = corridor_level()
    route = np.asarray([[Action.NOOP] * 40 + [Action.SLEEP] + [Action.RIGHT] * 3 + [NO_ACTION] * 4])
    (info,) = engine.replay_batch([level], task, route, 0.05, 120)
    assert tuple(info["positions"][-1]) == (3, 1), "asleep: the moves did nothing"
    assert info["wasted_actions"] == 44 and info["illegal_moves"] == 0, "40 NOOPs, SLEEP, 3 moves while asleep"


def test_replays_are_deterministic_under_the_fixed_key(demos):
    route = np.asarray([[Action.DO] * 20 + [NO_ACTION] * 44])  # DO on grass rolls a sapling on the rng
    level = demos.level(0)
    a = engine.replay_batch([level], demos.task, route, 0.05, 120)[0]
    b = engine.replay_batch([level], demos.task, route, 0.05, 120)[0]
    assert a["wasted_actions"] == b["wasted_actions"] == 20 and np.array_equal(a["positions"], b["positions"])


def test_worlds_must_be_square():
    walls = np.ones((5, 7), dtype=bool)
    walls[1:-1, 1:-1] = False
    level = Level(walls=walls, agent_start=(1, 1), objectives=(Objective((3, 5), 1.0, 0), Objective((1, 3), 0.5, 1)))
    with pytest.raises(ValueError, match="square"):
        engine.build_state(level, CraftaxTask())


def test_maze_demo_sets_still_replay_per_row(dataset):
    maze = DemoSet.generate(dataset, np.arange(4), rho=1.0)
    assert not hasattr(maze, "replay_many")
    info = maze.replay(0, maze.routes([0])[0])
    assert info["reached_index"] == maze.target[0]
