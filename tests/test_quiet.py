"""Tests for the gymnasium deprecation filter.

The filter earns its place by keeping ``results/`` legible, so what matters is
that it catches every message that was actually drowning those files and lets
anything else through. The messages are pinned here verbatim: a gymnasium
upgrade that rewords them should fail this rather than quietly refill the
results with 7 MB of warnings.
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import pytest

from goalmisgen import quiet

SILENCED = [
    "`gymnasium.vector.make(...)` is deprecated and will be replaced by `gymnasium.make_vec(...)` in v1.0",
    "env.max_episode_steps to get variables from other wrappers is deprecated and will be removed in v1.0, "
    "to get this variable you can do `env.unwrapped.max_episode_steps` for environment variables.",
    "env.num_envs to get variables from other wrappers is deprecated and will be removed in v1.0, "
    "to get this variable you can do `env.unwrapped.num_envs` for environment variables.",
    "env.observation_space to get variables from other wrappers is deprecated and will be removed in v1.0, "
    "to get this variable you can do `env.unwrapped.observation_space` for environment variables.",
    "env.single_action_space to get variables from other wrappers is deprecated and will be removed in v1.0, "
    "to get this variable you can do `env.unwrapped.single_action_space` for environment variables.",
]
"""Every distinct warning found in the results corpus, as gymnasium phrases it."""


def warnings_from(message: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet.install()
        gym.logger.warn(message)
    return list(caught)


@pytest.mark.parametrize("message", SILENCED)
def test_the_messages_that_drowned_the_results_are_filtered(message: str) -> None:
    assert warnings_from(message) == []


def test_a_warning_about_our_own_usage_still_gets_through() -> None:
    """The reason for filtering on the phrasing rather than raising the log level."""
    caught = warnings_from("The obs returned by the `reset()` method is not within the observation space")
    assert len(caught) == 1


def test_installed_on_importing_the_package() -> None:
    import goalmisgen

    assert goalmisgen.quiet is quiet
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        goalmisgen.quiet.install()
        gym.logger.warn(SILENCED[0])
    assert caught == []
