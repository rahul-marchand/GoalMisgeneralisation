"""Receding horizon: suffix states match the engine, and the closed loop reproduces the open loop."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.craftax import closed_loop, engine, simulate
from goalmisgen.craftax.craft import STATE_PLANES, CraftDemoSet, FieldSampler, SuffixDemoSet
from goalmisgen.offline.decode import evaluate, replay_all
from goalmisgen.offline.demonstrations import PROTOCOL_ATTRIBUTES
from goalmisgen.offline.demos import NO_ACTION
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.train import initial_params

N = 24


@pytest.fixture(scope="module")
def demos() -> CraftDemoSet:
    return CraftDemoSet.generate(FieldSampler(), seed=3, start=0, count=N, rho=1.0).with_hidden_values()


@pytest.fixture(scope="module")
def suffixes(demos) -> SuffixDemoSet:
    return demos.suffixes()


def test_suffix_set_is_one_item_per_state(demos, suffixes):
    assert len(suffixes) == int(np.asarray(demos.lengths).sum())
    assert suffixes.n_channels == demos.n_channels == 4 + 2 + STATE_PLANES
    missing = [n for n in PROTOCOL_ATTRIBUTES if not hasattr(suffixes, n)]
    assert not missing
    # item 0 is field 0 at t=0: the same observation and route as the whole set
    assert np.array_equal(suffixes.observations([0]), demos.observations([0]))
    assert np.array_equal(suffixes.routes([0]), demos.routes([0]))
    # the last item of field 0 is its final DO
    last = int(demos.lengths[0]) - 1
    assert suffixes.field_of[last] == 0 and suffixes.t_of[last] == last
    route = suffixes.routes([last])[0]
    assert route[0] == 5 and (route[1:] == NO_ACTION).all() and suffixes.lengths[last] == 1


def test_suffix_observations_are_what_the_engine_shows_at_that_step(demos, suffixes):
    task = demos.task
    rng = np.random.default_rng(0)
    items = rng.choice(len(suffixes), size=8, replace=False)
    observations = suffixes.observations(items)
    for row, j in enumerate(items):
        i, t = int(suffixes.field_of[j]), int(suffixes.t_of[j])
        field = demos.level(i)
        prefix = np.full(max(t, 1), NO_ACTION, dtype=np.int32)
        prefix[:t] = demos.routes([i])[0][:t]
        rollout = engine.run(engine.stack_states([engine.build_state(field, task)]), task, prefix[None], step_limit=200)
        state = rollout.final
        rendered = task.observe(
            np.asarray(state.map[0]),
            tuple(int(v) for v in np.asarray(state.player_position[0])),
            demos.feature_values([i])[0],
            True,
            int(state.player_direction[0]),
            [int(getattr(state.inventory, f)[0]) for f in simulate.INVENTORY],
        )
        assert np.array_equal(observations[row], rendered), (i, t)


def test_closed_loop_driven_by_the_expert_reproduces_the_open_loop(demos):
    indices = np.arange(N)
    routes = demos.routes(indices)
    model = RoutePrefixLM(
        ModelConfig(size=15, n_channels=demos.n_channels, n_actions=17, max_actions=128, d_model=32, n_layers=1, n_heads=1)
    )
    params = initial_params(model, __import__("jax").random.PRNGKey(0))

    def expert(observations, t):
        return np.where(routes[:, t] >= 0, routes[:, t], model.config.eos)

    decoded = closed_loop.rollout(model, params, demos, indices, policy=expert)
    assert np.array_equal(decoded.lengths, np.asarray(demos.lengths))
    for i in indices:
        n = int(demos.lengths[i])
        assert np.array_equal(decoded.actions[i, :n], routes[i, :n])
    outcomes = replay_all(demos, indices, decoded)
    assert all(o["reached_objective"] and o["reached_index"] == demos.target[i] for i, o in enumerate(outcomes))


def test_evaluate_dispatches_to_the_closed_loop_for_a_receding_task(demos):
    model = RoutePrefixLM(
        ModelConfig(size=15, n_channels=demos.n_channels, n_actions=17, max_actions=128, d_model=32, n_layers=1, n_heads=1)
    )
    params = initial_params(model, __import__("jax").random.PRNGKey(0))
    summary, decoded, outcomes = evaluate(model, params, demos, np.arange(6))
    assert len(outcomes) == 6 and decoded.actions.shape == (6, 128)
    assert 0.0 <= summary.behaviour.reached_objective <= 1.0
    with pytest.raises(ValueError, match="closed-loop"):
        evaluate(model, params, demos, np.arange(2), edit=np.zeros((2, 225, 32), dtype=np.float32))


def test_an_open_loop_crafting_task_still_decodes_open_loop():
    from goalmisgen.craftax.craft import CraftTask

    task = CraftTask(receding=False)
    demos = CraftDemoSet.generate(FieldSampler(), seed=4, start=0, count=3, rho=1.0, task=task)
    assert demos.decode_closed_loop is None and demos.n_channels == 4 + 2 + 1


def test_the_ineffective_repeat_guard_breaks_a_no_op_loop(demos):
    """A policy that always crafts a stone pickaxe loops forever; with the guard it is pushed to its next choice."""
    from goalmisgen.craftax.blocks import Action

    model = RoutePrefixLM(
        ModelConfig(size=15, n_channels=demos.n_channels, n_actions=17, max_actions=32, d_model=32, n_layers=1, n_heads=1)
    )
    params = initial_params(model, __import__("jax").random.PRNGKey(0))
    # Bias the untrained model's first-token logits so MAKE_STONE_PICKAXE always wins: patch the head bias.
    import jax

    flat, unravel = jax.flatten_util.ravel_pytree(params)
    head = [
        k
        for k in jax.tree_util.tree_leaves_with_path(params)
        if "head" in jax.tree_util.keystr(k[0]) and "bias" in jax.tree_util.keystr(k[0])
    ]
    assert head, "expected a head bias"
    biased = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf.at[int(Action.MAKE_STONE_PICKAXE)].set(50.0)
        if ("head" in jax.tree_util.keystr(path) and "bias" in jax.tree_util.keystr(path))
        else leaf,
        params,
    )
    plain = closed_loop.rollout(model, biased, demos, np.arange(4))
    guarded = closed_loop.rollout(model, biased, demos, np.arange(4), avoid_ineffective_repeat=True)
    assert (plain.actions[:, :5] == int(Action.MAKE_STONE_PICKAXE)).all(), "unguarded: the no-op repeats"
    assert (guarded.actions[:, 1] != int(Action.MAKE_STONE_PICKAXE)).all(), "guarded: after one no-op, something else"


def test_after_pickaxe_weight_repeats_only_the_late_states(demos):
    plain = demos.suffixes()
    heavy = demos.suffixes(after_pickaxe_weight=3)
    routes = demos.routes(np.arange(len(demos)))
    late = 0
    for i in range(len(demos)):
        r = routes[i].tolist()
        k = r.index(11)  # MAKE_WOOD_PICKAXE
        late += int(demos.lengths[i]) - (k + 1)
    assert len(heavy) == len(plain) + 2 * late
    assert (heavy.t_of[len(plain) :] > 0).all()
    j = len(plain)
    assert np.array_equal(
        heavy.observations([j]),
        plain.observations([np.nonzero((plain.field_of == heavy.field_of[j]) & (plain.t_of == heavy.t_of[j]))[0][0]]),
    )
