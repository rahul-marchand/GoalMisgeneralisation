"""Counterfactual starting states: the engine, the closed loop and the expert agree on them."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.craftax import closed_loop, engine, simulate
from goalmisgen.craftax.craft import CraftDemoSet, FieldSampler
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.train import initial_params


@pytest.fixture(scope="module")
def demos() -> CraftDemoSet:
    return CraftDemoSet.generate(FieldSampler(), seed=6, start=0, count=12, rho=1.0).with_hidden_values()


def test_a_counterfactual_start_replays_through_the_engine_to_the_expert_plan_from_it(demos):
    task = demos.task
    fields = [demos.level(i) for i in range(12)]
    starts = [simulate.State(task.tiles(f), f.agent_start, task.start_direction, (0, 0, 1, 0)) for f in fields]
    solutions = [task.solution_from(s, f, 0.05, 200) for s, f in zip(starts, fields)]
    routes = np.full((12, 128), -1, dtype=np.int32)
    for row, (s, f, sol) in enumerate(zip(starts, fields, solutions)):
        plan = task.plans_from(s, f)[sol.optimal_index]
        routes[row, : plan.cost] = plan.actions
    states = engine.stack_states([engine.state_from_simulated(s) for s in starts])
    outcomes = engine.replay_batch(fields, task, routes, 0.05, 200, states=states, solutions=solutions)
    for o, sol, s in zip(outcomes, solutions, starts):
        assert o["reached_objective"] and o["reached_index"] == sol.optimal_index and o["chose_optimal"]
        assert o["episode_steps"] == sol.distances[sol.optimal_index]
        assert (
            o["optimal_distance"] == sol.distances[sol.optimal_index]
        ), "scored against the plan from the counterfactual state"


def test_closed_loop_from_counterfactual_starts_driven_by_the_expert(demos):
    task = demos.task
    fields = [demos.level(i) for i in range(12)]
    starts = [simulate.State(task.tiles(f), f.agent_start, task.start_direction, (2, 0, 0, 0)) for f in fields]
    routes = np.full((12, 128), -1, dtype=np.int32)
    for row, (s, f) in enumerate(zip(starts, fields)):
        sol = task.solution_from(s, f, 0.05, 200)
        plan = task.plans_from(s, f)[sol.optimal_index]
        routes[row, : plan.cost] = plan.actions
    model = RoutePrefixLM(
        ModelConfig(size=15, n_channels=demos.n_channels, n_actions=17, max_actions=128, d_model=32, n_layers=1, n_heads=1)
    )
    params = initial_params(model, __import__("jax").random.PRNGKey(0))
    decoded = closed_loop.rollout(
        model, params, demos, np.arange(12), starts=starts, policy=lambda obs, t: np.where(routes[:, t] >= 0, routes[:, t], 17)
    )
    for row in range(12):
        n = int((routes[row] >= 0).sum())
        assert decoded.lengths[row] == n and np.array_equal(decoded.actions[row, :n], routes[row, :n])
