"""Tests for how a configured seed reaches the vector environment.

Both halves matter and they pull against each other. The correlation arms must
see the same levels, or the gap between them carries level-difficulty variance
instead of the effect. But the level sequence must still advance across resets,
because cleanba's evaluator reaches its Nth batch of levels by resetting N
times — pin every reset and it silently scores one batch over and over.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.configs.env import MazeConfig

WALL_CHANNEL, AGENT_CHANNEL, VALUE_CHANNEL = 0, 1, 4


def batches(seed: int, correlation: float = 1.0, n: int = 3) -> list[np.ndarray]:
    config = MazeConfig(
        max_episode_steps=120,
        num_envs=4,
        min_size=11,
        max_size=11,
        randomise_values=True,
        feature_value_correlation=correlation,
        asynchronous=False,
        seed=seed,
    )
    envs = config.make()
    return [envs.reset()[0].copy() for _ in range(n)]


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
