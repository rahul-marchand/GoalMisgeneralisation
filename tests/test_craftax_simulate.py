"""The numpy simulator matches the engine along expert routes, step for step."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.craftax import engine, simulate
from goalmisgen.craftax.blocks import Action, Block
from goalmisgen.craftax.craft import CraftDemoSet, FieldSampler

N = 40


@pytest.fixture(scope="module")
def demos() -> CraftDemoSet:
    return CraftDemoSet.generate(FieldSampler(), seed=2, start=0, count=N, rho=1.0)


def engine_states(field, task, route):
    """Engine state after each valid action, via one rollout per prefix length (slow, exact)."""
    n = int((route >= 0).sum())
    out = []
    for t in range(n + 1):
        prefix = np.full(max(n, 1), -1, dtype=np.int32)
        prefix[:t] = route[:t]
        states = engine.stack_states([engine.build_state(field, task)])
        rollout = engine.run(states, task, prefix[None], step_limit=200)
        final = rollout.final
        inv = final.inventory
        out.append(
            (
                np.asarray(final.map[0]),
                tuple(int(v) for v in np.asarray(final.player_position[0])),
                int(final.player_direction[0]),
                tuple(int(getattr(inv, f)[0]) for f in simulate.INVENTORY),
            )
        )
    return out


def test_simulator_matches_the_engine_along_expert_routes(demos):
    task = demos.task
    for i in range(6):
        field = demos.level(i)
        route = demos.routes([i])[0]
        start = simulate.initial(task.tiles(field), field.agent_start, task.start_direction, task.tools)
        sim = simulate.trajectory(start, route)
        eng = engine_states(field, task, route)
        assert len(sim) == len(eng)
        for t, (s, (tiles, position, facing, inventory)) in enumerate(zip(sim, eng)):
            assert np.array_equal(s.tiles, tiles), (i, t)
            assert s.position == position and s.facing == facing, (i, t)
            assert s.inventory == inventory, (i, t)
        assert sim[-1].collected == task.kinds[field.objectives[int(demos.target[i])].feature_id]


def test_simulator_rules_by_hand():
    tiles = np.full((7, 7), int(Block.GRASS), dtype=np.int32)
    tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = int(Block.FURNACE)
    tiles[3, 5] = int(Block.TREE)
    tiles[1, 3] = int(Block.STONE)
    s = simulate.initial(tiles, (3, 3), int(Action.DOWN))
    s = simulate.step(s, Action.RIGHT)  # (3,4) facing right at the tree
    assert s.position == (3, 4) and s.facing == Action.RIGHT
    s = simulate.step(s, Action.RIGHT)  # blocked by the tree: turn only
    assert s.position == (3, 4)
    s = simulate.step(s, Action.DO)
    assert s.wood == 1 and s.tiles[3, 5] == Block.GRASS
    s = simulate.step(s, Action.PLACE_TABLE)  # needs two wood
    assert s.tiles[3, 5] == Block.GRASS and s.wood == 1
    s = simulate.step(s, Action.MAKE_WOOD_PICKAXE)  # no table nearby
    assert s.inventory[2] == 0
    s2 = simulate.step(simulate.step(s, Action.UP), Action.DO)  # facing grass: nothing
    assert s2.inventory == s.inventory and np.array_equal(s2.tiles, s.tiles)
    s3 = simulate.step(simulate.step(simulate.step(s, Action.UP), Action.UP), Action.DO)  # bedrock: nothing
    assert s3.position == (1, 4) and np.array_equal(s3.tiles, s.tiles)
