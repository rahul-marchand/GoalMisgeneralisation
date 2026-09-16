"""The crafting task against the engine: plans that the engine executes exactly.

The planner is optimal within its plan family (see the module docstring); what
these tests hold it to is that every plan it emits, run through the real
``craftax_step``, collects the target ore in exactly its cost, having crafted
the pickaxe the recipe needs, and that the stored observation is what the
engine renders.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from goalmisgen.craftax import engine
from goalmisgen.craftax.blocks import Action, Block
from goalmisgen.craftax.craft import (
    RECIPES,
    CraftDemoSet,
    CraftTask,
    Field,
    FieldSampler,
    demonstrate_block,
)
from goalmisgen.envs.level import Objective
from goalmisgen.envs.values import FixedValues
from goalmisgen.offline.demonstrations import PROTOCOL_ATTRIBUTES, Demonstrations, load_demonstrations
from goalmisgen.offline.demos import shared_levels

N = 60


@pytest.fixture(scope="module")
def task() -> CraftTask:
    return CraftTask()


@pytest.fixture(scope="module")
def demos(task) -> CraftDemoSet:
    return CraftDemoSet.generate(FieldSampler(), seed=0, start=0, count=N, rho=1.0, task=task, workers=1)


def test_craft_demos_carry_every_protocol_attribute(demos):
    missing = [name for name in PROTOCOL_ATTRIBUTES if not hasattr(demos, name)]
    assert not missing
    assert isinstance(demos, Demonstrations)
    assert demos.n_actions == 17 and demos.max_actions == 128 and demos.n_channels == 3 + 2 + 1


def test_every_plan_runs_in_the_engine_to_its_target_in_exactly_its_cost(demos):
    indices = np.arange(N)
    outcomes = demos.replay_many(indices, demos.routes(indices), [True] * N)
    for i, info in enumerate(outcomes):
        target = int(demos.target[i])
        assert info["reached_objective"] and info["reached_index"] == target, i
        assert info["episode_steps"] == demos.distances[i, target] == demos.lengths[i], i
        assert info["illegal_moves"] == 0 and info["wasted_actions"] == 0, i
        assert info["chose_optimal"] or info["is_ambiguous"], i


def test_plans_craft_the_tool_the_recipe_needs(demos, task):
    for i in range(N):
        field = demos.level(i)
        target = int(demos.target[i])
        kind = task.kinds[field.objectives[target].feature_id]
        recipe = RECIPES[kind]
        state = engine.stack_states([engine.build_state(field, task)])
        rollout = engine.run(state, task, demos.routes([i]), step_limit=200)
        inventory = rollout.final.inventory
        assert int(getattr(inventory, recipe.tool)[0]) == 1, (i, recipe)
        assert int(inventory.wood[0]) == 0, "every wood is spent"
        if recipe.stone:
            assert (
                int(inventory.stone[0]) == 0
                and int(rollout.final.map[0].__array__().flatten().tolist().count(int(Block.CRAFTING_TABLE))) == 1
            )
        route = demos.routes([i])[0]
        actions = route[route >= 0].tolist()
        assert actions.count(int(Action.PLACE_TABLE)) == 1 and actions.count(int(Action.MAKE_WOOD_PICKAXE)) == 1
        assert actions.count(int(Action.DO)) == recipe.wood + recipe.stone + 1


def test_both_objectives_are_costed_and_the_cheaper_utility_wins(demos, task):
    for i in range(10):
        field = demos.level(i)
        plans = task.plans(field)
        assert all(p is not None for p in plans)
        costs = [p.cost for p in plans]
        assert tuple(int(c) for c in demos.distances[i]) == tuple(costs)
        utilities = [o.value - 0.05 * c for o, c in zip(field.objectives, costs)]
        assert int(demos.target[i]) == int(np.argmax(utilities)) or demos.ambiguous[i]


def test_stored_observation_is_what_the_engine_renders(demos, task):
    for i in range(5):
        field = demos.level(i)
        values = [0.0] * task.n_features
        for objective in field.objectives:
            values[objective.feature_id] = objective.value
        state = engine.build_state(field, task)
        assert np.array_equal(engine.render(state, values, task), demos.observations([i])[0])
        assert np.array_equal(
            engine.render(state, values, task, hide_values=True), demos.with_hidden_values().observations([i])[0]
        )
    obs = demos.observations([0])[0]
    assert obs[..., 2].sum() == FieldSampler().n_trees, "one tree channel"


def test_pools_are_paired_across_rho_and_values(task):
    a = demonstrate_block(FieldSampler(), 3, 5, 9, 1.0, task, 0, 0.05, 200, 128)
    b = demonstrate_block(FieldSampler(), 3, 5, 9, 0.0, task, 0, 0.05, 200, 128)
    c = demonstrate_block(FieldSampler(values=FixedValues((1.4, 0.5))), 3, 5, 9, 1.0, task, 0, 0.05, 200, 128)
    assert np.array_equal(a["walls_packed"], b["walls_packed"]) and np.array_equal(a["positions"], b["positions"])
    assert np.array_equal(a["feature_ids"], 1 - b["feature_ids"]), "rho 0 is the reversed assignment"
    assert np.array_equal(a["walls_packed"], c["walls_packed"]) and np.array_equal(a["trees_packed"], c["trees_packed"])
    assert np.array_equal(a["level_index"], np.arange(5, 9))


def test_save_load_subset_and_shared_levels(demos, tmp_path):
    demos.save(tmp_path / "demos")
    loaded = load_demonstrations(tmp_path / "demos", hide_values=True)
    assert isinstance(loaded, CraftDemoSet) and loaded.hide_values and loaded.task == demos.task
    assert np.array_equal(loaded.routes([2]), demos.routes([2]))
    small = loaded.subset([1, 2, 3])
    assert len(small) == 3 and small.level(0).agent_start == loaded.level(1).agent_start
    other = CraftDemoSet.generate(FieldSampler(), seed=0, start=N - 5, count=10, rho=1.0)
    assert shared_levels(demos, other) == 5
    assert shared_levels(demos, CraftDemoSet.generate(FieldSampler(), seed=1, start=0, count=3, rho=1.0)) == 0


def test_generation_is_jax_free():
    probe = (
        "import sys; from goalmisgen.craftax import craft; assert 'jax' not in sys.modules and 'craftax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_field_validation():
    walls = np.ones((7, 7), dtype=bool)
    walls[1:-1, 1:-1] = False
    trees = np.zeros_like(walls)
    trees[2, 2] = True
    with pytest.raises(ValueError, match="not free"):
        Field(walls, trees, (2, 2), (Objective((1, 1), 1.0, 0), Objective((5, 5), 0.5, 1)))
    with pytest.raises(ValueError, match="both"):
        Field(walls, walls.copy(), (3, 3), (Objective((1, 1), 1.0, 0), Objective((5, 5), 0.5, 1)))
    with pytest.raises(ValueError, match="recipe"):
        CraftTask(kinds=(int(Block.DIAMOND), int(Block.COAL)))


def test_a_hand_built_field_plans_the_expected_chain(task):
    walls = np.ones((9, 9), dtype=bool)
    walls[1:-1, 1:-1] = False
    trees = np.zeros_like(walls)
    for cell in ((2, 4), (4, 6), (6, 4)):
        trees[cell] = True
    field = Field(walls, trees, (4, 4), (Objective((1, 7), 1.0, 0), Objective((7, 1), 0.5, 1)))
    coal_index = [i for i, o in enumerate(field.objectives) if task.kinds[o.feature_id] == int(Block.COAL)][0]
    plan = task.plan(field, coal_index)
    assert plan is not None
    names = [Action(a).name for a in plan.actions]
    assert names.count("DO") == 4 and names.index("PLACE_TABLE") == names.index("MAKE_WOOD_PICKAXE") - 1
    iron_index = 1 - coal_index
    assert task.plan(field, iron_index) is None, "only three trees: no stone pickaxe"
    (info,) = engine.replay_batch([field], task, np.asarray([list(plan.actions)]), 0.05, 200)
    assert info["reached_objective"] and info["reached_index"] == coal_index and info["episode_steps"] == plan.cost


def test_routes_are_canonical_forward_greedy():
    """Every move is the first, in the fixed order, that shortens the distance to the leg's target."""
    from goalmisgen.craftax.routes import ORDER, canonical_moves, distances_to
    from goalmisgen.envs.solver import MOVES

    walls = np.ones((9, 9), dtype=bool)
    walls[1:-1, 1:-1] = False
    walls[3, 2:6] = True
    moves = canonical_moves(walls, (1, 1), (5, 5))
    assert moves is not None and len(moves) == 8
    field = distances_to(walls, (5, 5))
    here = (1, 1)
    for move in moves:
        wanted = int(field[here]) - 1
        first = next(m for m in ORDER if field[here[0] + MOVES[m][0], here[1] + MOVES[m][1]] == wanted)
        assert move == first
        here = (here[0] + MOVES[move][0], here[1] + MOVES[move][1])
    assert here == (5, 5)
    assert canonical_moves(walls, (1, 1), (1, 1)) == []


def test_plans_are_a_function_of_the_map(task):
    """The same field always gets the same plan, and a translated field a translated plan."""
    field = CraftDemoSet.generate(FieldSampler(), seed=5, start=0, count=1, rho=1.0).level(0)
    a = task.plans(field)
    b = task.plans(field)
    assert a == b
