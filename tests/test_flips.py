"""Tests for the behavioural flip-point step fit."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.flips import step_fit

THETAS = np.array([2.0, 4.0, 6.0, 8.0, 10.0])


def test_clean_steps_flip_at_the_midpoint():
    chosen = np.array(
        [
            [0, 0, 1],  # theta 2
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],  # theta 10
        ],
        dtype=bool,
    )
    result = step_fit(THETAS, chosen)
    np.testing.assert_allclose(result.flip, [7.0, 5.0, np.nan], equal_nan=True)
    assert result.violations.tolist() == [0, 0, 0]
    assert result.censored_low.tolist() == [False, False, True]
    assert result.censored_high.tolist() == [False, False, False]


def test_censoring_both_ways():
    never = np.zeros((5, 1), dtype=bool)
    always = np.ones((5, 1), dtype=bool)
    low = step_fit(THETAS, always)
    high = step_fit(THETAS, never)
    assert low.censored_low[0] and not low.bracketed[0]
    assert high.censored_high[0] and not high.bracketed[0]
    assert low.violations[0] == 0 and high.violations[0] == 0


def test_violations_count_what_the_step_cannot_explain():
    # Rises at theta 6 but dips back at theta 8: one choice off a clean step.
    chosen = np.array([[0], [0], [1], [0], [1]], dtype=bool)
    result = step_fit(THETAS, chosen)
    assert result.violations[0] == 1
    assert np.isfinite(result.flip[0])


def test_flip_is_where_the_misclassifications_balance():
    rng = np.random.default_rng(0)
    thetas = np.sort(rng.uniform(1, 20, size=41))
    true_flip = rng.uniform(3, 18, size=200)
    chosen = thetas[:, None] > true_flip[None, :]
    result = step_fit(thetas, chosen)
    assert result.violations.max() == 0
    # Noiseless steps recover the flip to within one grid interval.
    bracketed = result.bracketed
    spacing = np.diff(thetas).max()
    assert np.all(np.abs(result.flip[bracketed] - true_flip[bracketed]) <= spacing)


def test_rejects_unsorted_thetas_and_bad_shapes():
    with pytest.raises(ValueError):
        step_fit(np.array([3.0, 1.0]), np.zeros((2, 4), dtype=bool))
    with pytest.raises(ValueError):
        step_fit(THETAS, np.zeros((4, 4), dtype=bool))
