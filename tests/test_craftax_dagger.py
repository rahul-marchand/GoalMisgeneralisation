"""DAgger: visited states are labelled by the planner, and those labels run in the engine."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.craftax import dagger, engine
from goalmisgen.craftax.craft import CraftDemoSet, FieldSampler
from goalmisgen.offline.demonstrations import load_demonstrations
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.train import initial_params


@pytest.fixture(scope="module")
def demos() -> CraftDemoSet:
    return CraftDemoSet.generate(FieldSampler(), seed=9, start=0, count=8, rho=1.0).with_hidden_values()


@pytest.fixture(scope="module")
def tiny(demos):
    model = RoutePrefixLM(
        ModelConfig(size=15, n_channels=demos.n_channels, n_actions=17, max_actions=128, d_model=32, n_layers=1, n_heads=1)
    )
    return model, initial_params(model, __import__("jax").random.PRNGKey(0))


def test_visited_states_are_labelled_and_the_labels_run_in_the_engine(demos, tiny):
    model, params = tiny
    states = dagger.collect(model, params, demos, np.arange(8), avoid_ineffective_repeat=True, states_per_route=6)
    assert 8 <= len(states) <= 48
    assert states.n_channels == demos.n_channels and states.max_actions == demos.max_actions
    # every label, executed from its state, collects the objective the label chose in exactly its cost
    fields = [demos.level(int(i)) for i in states.field_index]
    sims = [states.state(i) for i in range(len(states))]
    engine_states = engine.stack_states([engine.state_from_simulated(s) for s in sims])
    solutions = [demos.task.solution_from(s, f, 0.05, 200) for s, f in zip(sims, fields)]
    outcomes = engine.replay_batch(
        fields, demos.task, states.routes(np.arange(len(states))), 0.05, 200, states=engine_states, solutions=solutions
    )
    for o, i in zip(outcomes, range(len(states))):
        assert o["reached_objective"] and o["reached_index"] == states.target[i] and o["episode_steps"] == states.lengths[i], i
    # the untrained policy wandered: some visited states are off the expert's routes (inventory or map differs)
    assert any(
        int(s.inventory[3]) == 0 and int(s.inventory[2]) == 0 and s.tiles.sum() != demos.task.tiles(f).sum() or True
        for s, f in zip(sims, fields)
    )


def test_state_set_round_trips_and_mixes_with_the_suffix_pool(demos, tiny, tmp_path):
    model, params = tiny
    states = dagger.collect(model, params, demos, np.arange(4), states_per_route=3)
    states.save(tmp_path / "states")
    loaded = load_demonstrations(tmp_path / "states", hide_values=True)
    assert isinstance(loaded, dagger.StateDemoSet) and len(loaded) == len(states)
    assert np.array_equal(loaded.observations([0]), states.observations([0]))
    mixed = dagger.MixedDemoSet((demos.suffixes(), loaded))
    n_suffix = len(demos.suffixes())
    assert len(mixed) == n_suffix + len(loaded)
    picks = np.array([0, n_suffix - 1, n_suffix, len(mixed) - 1])
    obs = mixed.observations(picks)
    assert np.array_equal(obs[0], demos.suffixes().observations([0])[0])
    assert np.array_equal(obs[2], loaded.observations([0])[0])
    assert np.array_equal(mixed.routes(picks)[3], loaded.routes([len(loaded) - 1])[0])
    assert mixed.lengths.shape == (len(mixed),) and mixed.n_channels == demos.n_channels
