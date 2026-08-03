"""Tests for objective value schemes."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.values import FixedValues, UniformValues


def test_fixed_values_are_returned_unchanged():
    scheme = FixedValues((1.0, 0.5))
    assert scheme.sample(2, np.random.default_rng(0)) == (1.0, 0.5)


def test_fixed_values_are_identical_across_episodes():
    scheme = FixedValues((1.0, 0.5))
    first = scheme.sample(2, np.random.default_rng(0))
    second = scheme.sample(2, np.random.default_rng(1))
    assert first == second


def test_fixed_values_rejects_a_count_mismatch():
    with pytest.raises(ValueError, match="2 values but the sampler asked for 3"):
        FixedValues((1.0, 0.5)).sample(3, np.random.default_rng(0))


def test_fixed_values_rejects_empty():
    with pytest.raises(ValueError, match="at least one value"):
        FixedValues(())


def test_uniform_values_lie_in_range():
    scheme = UniformValues(low=0.25, high=1.0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        values = scheme.sample(3, rng)
        assert len(values) == 3
        assert all(0.25 <= value <= 1.0 for value in values)


def test_uniform_values_vary_across_episodes():
    scheme = UniformValues()
    rng = np.random.default_rng(0)
    assert scheme.sample(2, rng) != scheme.sample(2, rng)


def test_uniform_values_are_deterministic_given_a_seed():
    scheme = UniformValues()
    assert scheme.sample(2, np.random.default_rng(3)) == scheme.sample(2, np.random.default_rng(3))


def test_uniform_values_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="low < high"):
        UniformValues(low=1.0, high=0.5)
