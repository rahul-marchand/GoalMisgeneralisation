"""Tests for the route model, its decoding, and a CPU smoke training run.

The mask test is the one that protects the experiment: if a cell token could
see an action token, "probe the residual at this cell before the first action"
would be reading something the actions had already written, and the DRC
analogy would be false without any number looking wrong.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goalmisgen.configs.env import MazeConfig
from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.decode import Decoded, evaluate, greedy_decode, replay
from goalmisgen.offline.demos import NO_ACTION, DemoSet
from goalmisgen.offline.model import (
    ModelConfig,
    RoutePrefixLM,
    cross_entropy,
    prefix_mask,
    targets_from_routes,
)
from goalmisgen.offline.train import TrainConfig, checkpoint_schedule, list_checkpoints, load_checkpoint, train

TINY = ModelConfig(size=7, n_channels=5, max_actions=16, d_model=32, n_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=300, seed=0, block_size=100)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16)


@pytest.fixture(scope="module")
def many_demos() -> DemoSet:
    """Enough 7x7 levels for a short CPU run to learn something general."""
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=3000, seed=1, block_size=1000)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16, workers=2)


def init(model, demos, n=4):
    observations = jnp.asarray(demos.observations(np.arange(n)))
    actions = jnp.asarray(demos.routes(np.arange(n)))
    params = model.init(jax.random.PRNGKey(0), observations, actions)
    return params, observations, actions


def test_prefix_mask_is_bidirectional_then_causal():
    mask = prefix_mask(3, 6)
    assert mask[:3, :3].all(), "prefix sees the whole prefix"
    assert not mask[:3, 3:].any(), "prefix never sees an action"
    assert mask[4, 3] and mask[4, 4] and not mask[4, 5], "actions are causal"


def test_cell_residuals_do_not_depend_on_the_actions(demos):
    model = RoutePrefixLM(TINY)
    params, observations, actions = init(model, demos)
    _, with_actions = model.apply(params, observations, actions)
    _, without = model.apply(params, observations, jnp.full_like(actions, NO_ACTION))
    for a, b in zip(with_actions, without):
        np.testing.assert_allclose(np.asarray(a[:, : TINY.n_cells]), np.asarray(b[:, : TINY.n_cells]), atol=1e-5)


def test_a_prediction_depends_only_on_earlier_actions(demos):
    model = RoutePrefixLM(TINY)
    params, observations, actions = init(model, demos)
    logits, _ = model.apply(params, observations, actions)
    changed = actions.at[:, 5].set((actions[:, 5] + 1) % 4)
    logits_changed, _ = model.apply(params, observations, changed)
    np.testing.assert_allclose(np.asarray(logits[:, :6]), np.asarray(logits_changed[:, :6]), atol=1e-5)
    assert not np.allclose(np.asarray(logits[:, 6]), np.asarray(logits_changed[:, 6]))


def test_shapes_and_targets(demos):
    model = RoutePrefixLM(TINY)
    params, observations, actions = init(model, demos)
    logits, residuals = model.apply(params, observations, actions)
    assert logits.shape == (4, TINY.max_actions + 1, TINY.n_classes)
    assert len(residuals) == TINY.n_layers + 1
    assert residuals[0].shape == (4, TINY.sequence_length, TINY.d_model)

    lengths = jnp.asarray(demos.lengths[:4], dtype=jnp.int32)
    targets = targets_from_routes(actions, lengths, TINY.eos)
    for row in range(4):
        n = int(lengths[row])
        assert np.array_equal(np.asarray(targets[row, :n]), np.asarray(actions[row, :n]))
        assert int(targets[row, n]) == TINY.eos
        assert np.all(np.asarray(targets[row, n + 1 :]) == -1)
    assert float(cross_entropy(logits, targets)) > 0


def test_loss_ignores_padding():
    logits = jnp.zeros((1, 3, 5)).at[0, 2, 0].set(50.0)  # a confident, wrong prediction at a padded position
    targets = jnp.array([[1, 4, -1]])
    assert float(cross_entropy(logits, targets)) == pytest.approx(np.log(5.0))


def test_replay_matches_the_environment(demos):
    """The replay must be the environment, or the numbers are not comparable."""
    from goalmisgen.envs.maze import MazeEnv
    from goalmisgen.envs.observation import ObservationEncoder

    rng = np.random.default_rng(0)
    for index in range(20):
        level = demos.level(index)
        actions = rng.integers(0, 4, size=rng.integers(1, 15))  # random walks hit walls and rarely arrive

        class Fixed:
            def sample(self, rng):
                return level

        env = MazeEnv(sampler=Fixed(), encoder=ObservationEncoder(max_size=7), step_penalty=0.05, step_limit=12)
        env.reset(seed=0)
        env_info = None
        for action in actions:
            _, _, terminated, truncated, env_info = env.step(int(action))
            if terminated or truncated:
                break
        assert env_info is not None
        ours = replay(level, actions, step_penalty=0.05, step_limit=12)
        for key, value in env_info.items():
            assert ours[key] == value, f"{key}: replay {ours[key]!r} != env {value!r}"


def test_replay_of_the_expert_route_is_optimal_and_legal(demos):
    for index in range(30):
        info = replay(demos.level(index), demos.actions[index], 0.05, 120)
        assert info["reached_objective"] and info["chose_optimal"] and info["illegal_moves"] == 0
        assert info["episode_steps"] == demos.lengths[index]
        assert info["visited"].sum() == demos.lengths[index] + 1


def test_replay_stops_at_eos_without_reaching():
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=1, seed=1, block_size=1)
    demos = DemoSet.generate(dataset, np.arange(1), rho=1.0, seed=0, max_actions=16)
    info = replay(demos.level(0), [], 0.05, 120)
    assert not info["reached_objective"] and info["episode_steps"] == 0


def test_checkpoint_schedule_is_dense_early_and_ends_on_the_last_step():
    steps = checkpoint_schedule(1000, first=10, ratio=2.0)
    assert steps[0] == 0 and steps[-1] == 1000
    assert steps[1:5] == (10, 20, 40, 80)
    assert len(steps) < 12


def test_untrained_decoding_has_the_right_shape(demos):
    model = RoutePrefixLM(TINY)
    params, _, _ = init(model, demos)
    decoded = greedy_decode(model, params, demos.observations(np.arange(6)), batch_size=4)
    assert isinstance(decoded, Decoded)
    assert decoded.actions.shape == (6, TINY.max_actions)
    assert np.all((decoded.actions == NO_ACTION) | ((decoded.actions >= 0) & (decoded.actions < 4)))
    for row in range(6):
        n = decoded.lengths[row]
        assert np.all(decoded.actions[row, :n] >= 0) and np.all(decoded.actions[row, n:] == NO_ACTION)


@pytest.mark.slow
def test_smoke_training_learns_legal_routes_on_cpu(many_demos, tmp_path):
    """Loss falls and greedy routes become legal and mostly reach the goal.

    Marked slow: a thousand steps on CPU take a minute or two. The bar is
    deliberately modest - this proves the pieces fit, not that the model is
    good - but "reaches the objective on levels it trained on, walks into no
    walls most of the time, and ends its own routes" is what a working trainer
    shows and a broken one does not. Held-out performance is printed, not
    asserted: at this scale it is a property of the budget, not of the code.
    """
    demos = many_demos
    held_out = np.arange(2800, 3000)
    seen = np.arange(200)
    train_set = demos.subset(np.arange(2800))
    config = TrainConfig(
        total_steps=1000,
        batch_size=64,
        learning_rate=3e-3,
        warmup_steps=50,
        log_every=100,
        checkpoint_first=50,
        checkpoint_ratio=4.0,
    )
    rows = []

    def evaluator(params, step):
        summary, _, _ = evaluate(RoutePrefixLM(TINY), params, demos, held_out)
        rows.append((step, summary))
        return summary.as_row()

    params = train(train_set, TINY, config, tmp_path / "run", evaluate=evaluator, log=lambda s: None)

    steps = [step for step, _ in list_checkpoints(tmp_path / "run")]
    assert steps[0] == 0 and steps[-1] == 1000
    metrics = (tmp_path / "run" / "metrics.csv").read_text().splitlines()
    first, last = float(metrics[1].split(",")[1]), float(metrics[-1].split(",")[1])
    assert last < 0.7 * first, f"loss did not fall: {first:.3f} -> {last:.3f}"

    untrained, trained = rows[0][1], rows[-1][1]
    model = RoutePrefixLM(TINY)
    on_seen, _, _ = evaluate(model, params, demos, seen)
    print(f"\nuntrained (held out): {untrained}\ntrained (held out):   {trained}\ntrained (seen):       {on_seen}")
    assert on_seen.behaviour.reached_objective > 0.5
    assert on_seen.legal > 0.5
    assert on_seen.emitted_eos > 0.9
    assert trained.behaviour.reached_objective > untrained.behaviour.reached_objective

    _, loaded = load_checkpoint(list_checkpoints(tmp_path / "run")[-1][1])
    observations = demos.observations(held_out[:8])
    a = greedy_decode(model, loaded, observations)
    b = greedy_decode(model, params, observations)
    np.testing.assert_array_equal(a.actions, b.actions)


def test_maze_config_channels_match_the_model_default():
    """The default ModelConfig must describe the 11x11 observation every DRC agent saw."""
    encoder = MazeConfig(max_episode_steps=120, min_size=11, max_size=11).encoder()
    assert encoder.shape == (ModelConfig().size, ModelConfig().size, ModelConfig().n_channels)
