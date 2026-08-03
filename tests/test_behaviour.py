"""Tests for behavioural measurement.

The autoreset test is the one that matters. A gymnasium vector environment
returns the *next* episode's info at a terminating step, so reading outcomes
from the top level misattributes every episode by one — silently, with
plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis import collect_episode_outcomes, summarise
from goalmisgen.configs.env import MazeConfig


def random_policy(rng):
    def policy(observations):
        return rng.integers(0, 4, size=len(observations))

    return policy


def test_outcomes_come_from_the_finished_episode_not_the_next_one():
    """final_info describes the episode that ended; top-level info does not."""
    config = MazeConfig(max_episode_steps=20, num_envs=8, min_size=5, max_size=5, asynchronous=False)
    envs = config.make()

    outcomes = collect_episode_outcomes(envs, random_policy(np.random.default_rng(0)), n_episodes=30, seed=0)
    assert len(outcomes) == 30
    # Every outcome is a completed episode, so it carries an episode length.
    assert all("episode_steps" in o for o in outcomes)
    assert all(o["episode_steps"] >= 1 for o in outcomes)
    # And the ground truth that came with that same level.
    assert all("optimal_index" in o for o in outcomes)


def test_summary_excludes_ambiguous_levels_from_optimality():
    """Either choice is optimal on a tie, so those episodes would inflate
    accuracy with coin flips."""
    outcomes = [
        {"reached_objective": True, "chose_optimal": True, "is_ambiguous": True, "reached_feature_id": 0},
        {"reached_objective": True, "chose_optimal": False, "is_ambiguous": False, "reached_feature_id": 1},
    ]
    summary = summarise(outcomes)
    assert summary.chose_optimal == 0.0, "the ambiguous episode must not count as a success"
    assert summary.ambiguous == 0.5


def test_summary_reports_proxy_following_separately_from_optimality():
    outcomes = [
        {"reached_objective": True, "chose_optimal": False, "is_ambiguous": False, "reached_feature_id": 0} for _ in range(4)
    ]
    summary = summarise(outcomes)
    assert summary.followed_feature_zero == 1.0
    assert summary.chose_optimal == 0.0


def test_timeouts_count_against_reaching_but_not_against_optimality():
    outcomes = [
        {"reached_objective": False, "episode_steps": 20},
        {"reached_objective": True, "chose_optimal": True, "is_ambiguous": False, "reached_feature_id": 0},
    ]
    summary = summarise(outcomes)
    assert summary.reached_objective == 0.5
    assert summary.chose_optimal == 1.0


def test_summarising_nothing_is_an_error():
    with pytest.raises(ValueError, match="no episodes"):
        summarise([])
