"""Tests for how a configured seed reaches the vector environment.

Both halves matter and they pull against each other. The correlation arms must
see the same levels, or the gap between them carries level-difficulty variance
instead of the effect. But the level sequence must still advance across resets,
because cleanba's evaluator reaches its Nth batch of levels by resetting N
times — pin every reset and it silently scores one batch over and over.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.configs.env import MazeConfig
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL, ObservationEncoder

VALUE_CHANNEL = ObservationEncoder(max_size=11, n_features=2).first_value_channel


def batches(seed: int, correlation: float = 1.0, n: int = 3, asynchronous: bool = False) -> list[np.ndarray]:
    config = MazeConfig(
        max_episode_steps=120,
        num_envs=4,
        min_size=11,
        max_size=11,
        asynchronous=asynchronous,
        randomise_values=True,
        feature_value_correlation=correlation,
        seed=seed,
    )
    envs = config.make()
    try:
        return [envs.reset()[0].copy() for _ in range(n)]
    finally:
        envs.close()


def test_successive_resets_draw_new_levels():
    """cleanba's evaluator advances through level batches by resetting again."""
    first, second, third = batches(seed=1)
    assert not np.array_equal(first, second)
    assert not np.array_equal(second, third)


def test_the_seed_makes_the_whole_sequence_reproducible():
    assert all(np.array_equal(a, b) for a, b in zip(batches(seed=1), batches(seed=1)))
    assert not np.array_equal(batches(seed=1)[0], batches(seed=2)[0])


def test_correlation_arms_are_scored_on_the_same_levels():
    """Only which objective wears feature 0 may differ between arms."""
    high, low = batches(seed=1, correlation=1.0), batches(seed=1, correlation=0.0)
    for channel in (WALL_CHANNEL, AGENT_CHANNEL, VALUE_CHANNEL):
        assert all(np.array_equal(a[:, channel], b[:, channel]) for a, b in zip(high, low))

    # Without this the test would pass even if the correlation were ignored
    # entirely, which is the failure that would make the experiment vacuous.
    assert not all(np.array_equal(a[:, FIRST_FEATURE_CHANNEL], b[:, FIRST_FEATURE_CHANNEL]) for a, b in zip(high, low))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_the_seed_reaches_an_asynchronous_env_too(asynchronous):
    """A synchronous env takes the seed in ``reset_wait``, an async one in
    ``reset_async``. Intercepting one left training envs, which default to
    asynchronous, drawing their levels from OS entropy."""
    first = batches(seed=5, n=1, asynchronous=asynchronous)
    again = batches(seed=5, n=1, asynchronous=asynchronous)
    other = batches(seed=6, n=1, asynchronous=asynchronous)

    assert np.array_equal(first[0], again[0]), "the same seed must reproduce the same levels"
    assert not np.array_equal(first[0], other[0]), "a different seed must give different levels"


def test_dropping_the_value_channel_needs_an_explicit_opt_in():
    """Without a value channel colour is the only cue to value, so a
    colour-follower and a value-follower are indistinguishable and no
    misgeneralisation can be measured. Legitimate for a mechanism experiment,
    never something to reach by accident."""
    with pytest.raises(ValueError, match="colour_is_the_only_value_cue"):
        MazeConfig(max_episode_steps=120, n_objectives=2, value_encoding="none")

    config = MazeConfig(max_episode_steps=120, n_objectives=2, value_encoding="none", colour_is_the_only_value_cue=True)
    encoder = config.encoder()
    assert encoder.n_value_channels == 0
    assert encoder.n_channels == 4, "walls, agent and two feature channels, and nothing else"
