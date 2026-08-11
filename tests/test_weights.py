"""Tests for the weight-diff comparison.

The whole value-axis experiment reduces to "do these diffs lie on a line", so a
fit that reports a line where there is none, or that quietly rescales one, would
manufacture the result it is meant to test. The negative cases matter more than
the positive one here.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.weights import cosine, explained, fit_axis, fit_axis_and_drift, projected_offset


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


def test_drift_shared_by_every_arm_is_not_mistaken_for_an_axis() -> None:
    """The failure the origin-forced fit actually produced.

    Give every arm a large common component and a small value-specific one, on
    offsets that are not balanced around zero. Forcing through the origin drags
    the drift into the axis, so it reads back nearly the same offset for every
    arm. Fitting an intercept recovers the real axis instead.
    """
    rng = np.random.default_rng(6)
    offsets = np.array([-0.2, -0.1, 0.1, 0.2, 0.3, 0.4])
    axis = rng.normal(size=256)
    drift = 20 * rng.normal(size=256)
    diffs = drift + np.outer(offsets, axis)

    naive = fit_axis(offsets, diffs)
    implied = np.array([projected_offset(d, naive) for d in diffs])
    assert implied.std() < 0.1 * offsets.std()  # every arm looks the same

    fitted, recovered_drift = fit_axis_and_drift(offsets, diffs)
    assert cosine(fitted, axis) == pytest.approx(1.0, abs=1e-8)
    assert cosine(recovered_drift, drift) == pytest.approx(1.0, abs=1e-8)


def test_intercept_fit_needs_three_arms() -> None:
    with pytest.raises(ValueError, match="at least three arms"):
        fit_axis_and_drift(np.array([-0.1, 0.1]), np.ones((2, 4)))


def test_symmetric_pairs_are_found_at_whatever_offsets_a_grid_uses() -> None:
    """The reliability estimate must not depend on a grid's particular offsets.

    Written-out pairs reported nothing the moment a grid was widened, which is
    when reliability matters most: the widening is done *because* reliability
    was too low to read.
    """
    for offsets in ([-0.4, -0.2, 0.2, 0.4], [-0.25, -0.1, 0.1, 0.25], [-0.2, -0.1, 0.1, 0.2]):
        table = {round(o, 3): None for o in offsets}
        pairs = [(m, -m) for m in sorted({abs(o) for o in table}, reverse=True) if m in table and -m in table]
        assert len(pairs) == 2, (offsets, pairs)
        assert pairs[0][0] > pairs[1][0], "widest pair first"


def test_an_asymmetric_grid_yields_too_few_pairs_to_estimate_reliability() -> None:
    offsets = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
    table = {round(o, 3): None for o in offsets}
    pairs = [(m, -m) for m in sorted({abs(o) for o in table}, reverse=True) if m in table and -m in table]
    assert len(pairs) == 2  # 0.2 and 0.1 pair; 0.3 and 0.4 have no partner
