"""Tests for the weight-diff comparison.

The whole value-axis experiment reduces to "do these diffs lie on a line", so a
fit that reports a line where there is none, or that quietly rescales one, would
manufacture the result it is meant to test. The negative cases matter more than
the positive one here.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.weights import cosine, explained, fit_axis, projected_offset


def linear_family(rng: np.random.Generator, offsets: np.ndarray, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Diffs that genuinely lie on one axis, optionally perturbed off it."""
    axis = rng.normal(size=64)
    diffs = np.outer(offsets, axis) + noise * rng.normal(size=(len(offsets), 64))
    return axis, diffs


def test_recovers_the_axis_a_family_was_built_from() -> None:
    rng = np.random.default_rng(0)
    offsets = np.array([-0.2, -0.1, 0.1, 0.2, 0.4])
    axis, diffs = linear_family(rng, offsets)

    fitted = fit_axis(offsets, diffs)

    assert cosine(fitted, axis) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(fitted, axis)


def test_axis_carries_scale_not_just_direction() -> None:
    """Doubling every diff must double the axis, or writing an offset means nothing."""
    rng = np.random.default_rng(1)
    offsets = np.array([-0.2, 0.1, 0.3])
    _, diffs = linear_family(rng, offsets)

    assert np.allclose(fit_axis(offsets, 2 * diffs), 2 * fit_axis(offsets, diffs))


def test_explained_is_one_on_the_axis_and_negative_off_it() -> None:
    rng = np.random.default_rng(2)
    offsets = np.array([-0.2, 0.1, 0.3])
    axis, diffs = linear_family(rng, offsets)

    assert explained(diffs[1], offsets[1], axis) == pytest.approx(1.0, abs=1e-9)
    # Predicting the opposite of what the arm did is worse than predicting nothing.
    assert explained(diffs[1], -offsets[1], axis) < 0


def test_unrelated_diffs_do_not_fit_a_line() -> None:
    """The refuting case: independent directions must not survive being held out."""
    rng = np.random.default_rng(3)
    offsets = np.array([-0.2, -0.1, 0.1, 0.2, 0.4])
    diffs = rng.normal(size=(len(offsets), 64))

    for held in range(len(offsets)):
        keep = [i for i in range(len(offsets)) if i != held]
        axis = fit_axis(offsets[keep], diffs[keep])
        assert abs(cosine(diffs[held], axis)) < 0.5
        assert explained(diffs[held], offsets[held], axis) < 0.5


def test_the_widest_arm_flatters_itself_in_sample() -> None:
    """Why the leave-one-out numbers are the ones that count.

    Least squares through the origin is dominated by the arm with the largest
    offset, so that arm is partly fitting itself: its in-sample score comes out
    high even when the diffs are independent noise with no axis to find. Reported
    in-sample, that single number would look like the result the experiment is
    trying to establish.
    """
    rng = np.random.default_rng(3)
    offsets = np.array([-0.2, -0.1, 0.1, 0.2, 0.4])
    diffs = rng.normal(size=(len(offsets), 64))

    axis = fit_axis(offsets, diffs)
    in_sample = [explained(diff, offset, axis) for diff, offset in zip(diffs, offsets)]

    widest = int(np.argmax(np.abs(offsets)))
    assert in_sample[widest] > 0.5
    assert in_sample[widest] == max(in_sample)


def test_a_held_out_arm_recovers_its_own_offset() -> None:
    """Fit without one arm, and the axis should say how far that arm went."""
    rng = np.random.default_rng(4)
    offsets = np.array([-0.2, -0.1, 0.1, 0.2, 0.4])
    _, diffs = linear_family(rng, offsets, noise=0.01)

    for held in range(len(offsets)):
        keep = [i for i in range(len(offsets)) if i != held]
        axis = fit_axis(offsets[keep], diffs[keep])
        assert projected_offset(diffs[held], axis) == pytest.approx(offsets[held], abs=0.02)


def test_noise_degrades_the_fit_rather_than_breaking_it() -> None:
    rng = np.random.default_rng(5)
    offsets = np.array([-0.2, -0.1, 0.1, 0.2, 0.4])
    axis, diffs = linear_family(rng, offsets, noise=0.05)

    fitted = fit_axis(offsets, diffs)
    assert cosine(fitted, axis) > 0.9


def test_offsets_that_are_all_zero_are_refused() -> None:
    """Every arm at the base value is a mistake in the experiment, not a fit of zero."""
    with pytest.raises(ValueError, match="no offset"):
        fit_axis(np.zeros(3), np.ones((3, 4)))


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="one offset per diff"):
        fit_axis(np.array([0.1, 0.2]), np.ones((3, 4)))
