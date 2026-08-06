"""Tests for the maze environment.

Beyond the gymnasium API contract, two properties matter most:

* following the solver's optimal path must earn exactly the utility the solver
  predicted — this ties the environment's reward structure to the ground truth
  everything downstream is measured against;
* ground truth must never leak into the observation.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from goalmisgen.envs.level import Level, Objective
from goalmisgen.envs.maze import MazeEnv
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import path_to_objective, solve
from goalmisgen.envs.values import FixedValues

OPEN_ROOM = np.ones((5, 5), dtype=np.bool_)
OPEN_ROOM[1:4, 1:4] = False

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class FixedLevelSampler:
    """Returns the same level every episode, for deterministic tests."""

    def __init__(self, level: Level) -> None:
        self.level = level

    def sample(self, rng) -> Level:
        del rng
        return self.level


def fixed_level(objectives=None) -> Level:
    if objectives is None:
        objectives = (
            Objective((1, 3), value=1.0, feature_id=0),
            Objective((3, 1), value=0.5, feature_id=1),
        )
    return Level(walls=OPEN_ROOM, agent_start=(1, 1), objectives=objectives)


def make_env(level: Level | None = None, **kwargs) -> MazeEnv:
    level = level if level is not None else fixed_level()
    defaults = dict(
        sampler=FixedLevelSampler(level),
        encoder=ObservationEncoder(max_size=9, n_features=2),
        step_penalty=0.01,
        step_limit=50,
    )
    defaults.update(kwargs)
    return MazeEnv(**defaults)  # type: ignore[arg-type]


def path_to_actions(path) -> list[int]:
    from goalmisgen.envs.solver import MOVES

    deltas = {delta: index for index, delta in enumerate(MOVES)}
    return [deltas[(after[0] - before[0], after[1] - before[1])] for before, after in zip(path, path[1:])]


# --------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------


def test_passes_the_gymnasium_env_checker():
    env = MazeEnv(
        sampler=MazeLevelSampler(size_range=(5, 9)),
        encoder=ObservationEncoder(max_size=9, n_features=2),
    )
    check_env(env, skip_render_check=True)


def test_reset_returns_an_observation_in_the_declared_space():
    env = make_env()
    observation, info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    assert info["optimal_index"] == 0  # nearer and more valuable


def test_same_seed_gives_the_same_episode():
    sampler = MazeLevelSampler(size_range=(5, 9))
    encoder = ObservationEncoder(max_size=9, n_features=2)
    first, _ = MazeEnv(sampler=sampler, encoder=encoder).reset(seed=7)
    second, _ = MazeEnv(sampler=sampler, encoder=encoder).reset(seed=7)
    assert np.array_equal(first, second)


def test_rejects_an_invalid_action():
    env = make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="invalid action"):
        env.step(99)


def test_rejects_invalid_construction():
    with pytest.raises(ValueError, match="step_penalty"):
        make_env(step_penalty=-1.0)
    with pytest.raises(ValueError, match="step_limit"):
        make_env(step_limit=0)


# --------------------------------------------------------------------------
# Movement
# --------------------------------------------------------------------------


def test_each_action_moves_in_the_expected_direction():
    env = make_env()
    env.reset(seed=0)
    env.agent_position = (2, 2)

    for action, expected in ((UP, (1, 2)), (DOWN, (3, 2)), (LEFT, (2, 1)), (RIGHT, (2, 3))):
        env.agent_position = (2, 2)
        env.step(action)
        assert env.agent_position == expected


def test_walls_block_movement_but_still_cost_a_step():
    env = make_env()
    env.reset(seed=0)
    assert env.agent_position == (1, 1)

    _, reward, terminated, truncated, _ = env.step(UP)  # into the top wall
    assert env.agent_position == (1, 1)
    assert reward == pytest.approx(-0.01)
    assert not terminated and not truncated


def test_the_observation_follows_the_agent():
    env = make_env()
    observation, _ = env.reset(seed=0)
    agent_channel = observation[:, :, 1]
    assert agent_channel[1, 1] == 1.0

    observation, *_ = env.step(DOWN)
    assert observation[2, 1, 1] == 1.0
    assert observation[1, 1, 1] == 0.0


# --------------------------------------------------------------------------
# Reward and termination
# --------------------------------------------------------------------------


def test_reaching_an_objective_terminates_and_pays_its_value():
    env = make_env()
    env.reset(seed=0)
    env.step(RIGHT)  # (1,1) -> (1,2)
    _, reward, terminated, truncated, info = env.step(RIGHT)  # -> (1,3), objective 0

    assert terminated and not truncated
    assert reward == pytest.approx(1.0 - 0.01)
    assert info["reached_index"] == 0
    assert info["reached_feature_id"] == 0
    assert info["chose_optimal"]
    assert info["episode_steps"] == 2


@pytest.mark.parametrize("actions,step_limit", [((RIGHT, RIGHT), 20), ((RIGHT, LEFT, RIGHT), 3)])
def test_the_reported_return_is_the_rewards_actually_paid(actions, step_limit):
    """Both endings: reaching an objective, and running out of steps.

    The reported return is derived rather than accumulated, so nothing else
    would notice if the derivation and the reward drifted apart.
    """
    env = make_env(step_limit=step_limit)
    env.reset(seed=0)

    total = 0.0
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break

    assert terminated or truncated, "the episode must end for an outcome to be reported"
    assert info["episode_return"] == pytest.approx(total)


def test_reaching_the_worse_objective_is_recorded_as_suboptimal():
    env = make_env()
    env.reset(seed=0)
    env.step(DOWN)
    _, reward, terminated, _, info = env.step(DOWN)  # -> (3,1), objective 1

    assert terminated
    assert reward == pytest.approx(0.5 - 0.01)
    assert info["reached_feature_id"] == 1
    assert not info["chose_optimal"]


def test_reset_survives_levels_whose_objectives_all_exceed_the_step_limit():
    """The production size range, which no other test reaches.

    At 25x25 with a 120-step limit about 2% of levels have every objective
    further away than the episode is long. ``solve`` refuses those, and the
    exception used to escape ``reset`` — under autoreset that kills an actor
    mid-rollout, hundreds of steps into a run rather than at startup.
    """
    env = MazeEnv(
        sampler=MazeLevelSampler(size_range=(25, 25)),
        encoder=ObservationEncoder(max_size=25, n_features=2),
        step_penalty=0.05,
        step_limit=120,
    )
    for seed in range(300):
        _, info = env.reset(seed=seed)
        assert info["optimal_distance"] <= 120, "the chosen objective must be reachable in time"


def test_episodes_truncate_at_the_step_limit():
    env = make_env(step_limit=3)
    env.reset(seed=0)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(UP)  # bump the wall, go nowhere
        assert not terminated and not truncated

    _, _, terminated, truncated, info = env.step(UP)
    assert truncated and not terminated
    assert info["reached_objective"] is False
    assert info["episode_steps"] == 3


def test_optimal_play_earns_exactly_the_predicted_utility():
    """Ties the environment's reward to the solver's utility definition."""
    for seed in range(10):
        env = MazeEnv(
            sampler=MazeLevelSampler(size_range=(5, 11)),
            encoder=ObservationEncoder(max_size=11, n_features=2),
            step_penalty=0.02,
            step_limit=500,
        )
        env.reset(seed=seed)
        solution = solve(env.level, env.step_penalty)
        target = env.level.objectives[solution.optimal_index]

        # Must route around the other objective: stepping on it would end the
        # episode early.
        path = path_to_objective(env.level, solution.optimal_index)
        assert path is not None
        assert path[-1] == target.position

        total = 0.0
        terminated = False
        for action in path_to_actions(path):
            assert not terminated, "episode ended before reaching the intended objective"
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
        assert terminated and not truncated
        assert info["reached_index"] == solution.optimal_index
        assert total == pytest.approx(solution.utilities[solution.optimal_index])


def test_step_penalty_can_make_the_near_objective_optimal():
    """The environment agrees with the solver about which target wins."""
    level = fixed_level(
        (
            Objective((1, 2), value=1.0, feature_id=0),  # distance 1
            Objective((3, 3), value=2.0, feature_id=1),  # distance 4
        )
    )
    wide = ObservationEncoder(max_size=9, n_features=2, value_range=(0.0, 2.0))
    assert make_env(level, encoder=wide, step_penalty=0.1).reset(seed=0)[1]["optimal_index"] == 1
    assert make_env(level, encoder=wide, step_penalty=0.5).reset(seed=0)[1]["optimal_index"] == 0


# --------------------------------------------------------------------------
# Ground truth must not leak into the observation
# --------------------------------------------------------------------------


def test_ground_truth_does_not_leak_into_the_observation():
    """Changing step_penalty changes which objective is optimal, but not the maze.

    If the observation encoded the answer rather than the raw scene, these two
    observations would have to differ.
    """
    level = fixed_level(
        (
            Objective((1, 2), value=1.0, feature_id=0),
            Objective((3, 3), value=2.0, feature_id=1),
        )
    )
    wide = ObservationEncoder(max_size=9, n_features=2, value_range=(0.0, 2.0))
    lenient, info_lenient = make_env(level, encoder=wide, step_penalty=0.1).reset(seed=0)
    harsh, info_harsh = make_env(level, encoder=wide, step_penalty=0.5).reset(seed=0)

    assert info_lenient["optimal_index"] != info_harsh["optimal_index"]
    assert np.array_equal(lenient, harsh)


def test_info_carries_the_ground_truth_the_observation_lacks():
    env = make_env()
    _, info = env.reset(seed=0)
    for key in ("optimal_index", "optimal_feature_id", "optimal_distance", "utility_margin", "is_ambiguous"):
        assert key in info


def test_info_values_are_scalars_for_vector_env_batching():
    env = make_env()
    _, info = env.reset(seed=0)
    for key, value in info.items():
        assert isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)), key


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_render_returns_an_rgb_image():
    env = make_env(render_mode="rgb_array")
    env.reset(seed=0)
    image = env.render()
    assert image is not None
    assert image.dtype == np.uint8
    assert image.shape == (5 * 8, 5 * 8, 3)


def test_render_returns_none_without_a_render_mode():
    env = make_env()
    env.reset(seed=0)
    assert env.render() is None


def test_defaults_construct_a_usable_environment():
    """The zero-argument form must work, since gym.make may pass no kwargs."""
    env = MazeEnv()
    observation, info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    assert "optimal_index" in info


def test_single_objective_configuration():
    env = MazeEnv(
        sampler=MazeLevelSampler(size_range=(5, 9), n_objectives=1, values=FixedValues((1.0,))),
        encoder=ObservationEncoder(max_size=9, n_features=1, value_encoding="none"),
    )
    observation, info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    assert info["optimal_index"] == 0


def test_the_solver_and_the_environment_agree_about_what_a_move_is():
    """If these diverge, every optimal_index label is quietly wrong.

    Adding a no-op action or diagonals to the environment without telling the
    solver would change reachable distances with nothing to crash.
    """
    from goalmisgen.envs.solver import MOVES

    assert len(MOVES) == MazeEnv().action_space.n


def test_info_reports_both_objectives_not_only_the_optimal_one():
    """Asking how the agent traded value against distance needs both sides of
    the comparison it faced, not just the side that won."""
    from goalmisgen.envs.sampling import MazeLevelSampler
    from goalmisgen.envs.solver import objective_distances

    env = MazeEnv(
        MazeLevelSampler(size_range=(11, 11)),
        ObservationEncoder(max_size=11, n_features=2),
        step_penalty=0.05,
    )

    for seed in range(20):
        _, info = env.reset(seed=seed)
        distances = objective_distances(env.level)
        for index, objective in enumerate(env.level.objectives):
            assert info[f"feature_{objective.feature_id}_value"] == objective.value
            expected = -1 if distances[index] is None else distances[index]
            assert info[f"feature_{objective.feature_id}_distance"] == expected, (
                f"feature {objective.feature_id} distance disagrees with the solver"
            )
