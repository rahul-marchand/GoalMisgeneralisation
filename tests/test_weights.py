"""Tests for the weight-diff comparison.

The whole value-axis experiment reduces to "do these diffs lie on a line", so a
fit that reports a line where there is none, or that quietly rescales one, would
manufacture the result it is meant to test. The negative cases matter more than
the positive one here.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.weights import (
    cosine,
    explained,
    fit_axis,
    fit_axis_and_drift,
    gram_matrix,
    permutation_cosines,
    permutation_norms,
    permutation_p_value,
    project,
    projected_offset,
    split_half_reliability,
)


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


def test_permutation_null_is_not_centred_on_zero_when_the_diffs_share_drift() -> None:
    """The whole reason for a permutation null rather than assuming zero.

    Every arm carries the same common component, so two axes fitted from the same
    agent share structure whether or not value is represented. A null of zero
    would read that shared drift as evidence.
    """
    rng = np.random.default_rng(7)
    offsets = np.array([-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])
    drift = 40.0 * rng.normal(size=64)
    diffs = drift + np.outer(offsets, rng.normal(size=64)) + 0.5 * rng.normal(size=(len(offsets), 64))

    null = permutation_cosines(offsets, diffs, rng.normal(size=64), resamples=200, seed=1)

    assert null.std() > 0.05, "a degenerate null would make every observation look significant"


def test_a_real_axis_beats_its_own_permutation_null() -> None:
    rng = np.random.default_rng(3)
    offsets = np.array([-0.4, -0.3, -0.2, 0.2, 0.3, 0.4])
    axis, diffs = linear_family(rng, offsets, noise=0.3)

    observed = cosine(fit_axis_and_drift(offsets, diffs)[0], axis)
    null = permutation_cosines(offsets, diffs, axis, resamples=400, seed=2)

    assert permutation_p_value(observed, null, alternative="greater") < 0.01


def test_shuffled_data_does_not_beat_the_null() -> None:
    """Guards the test above: the procedure must not declare everything significant."""
    rng = np.random.default_rng(4)
    offsets = np.array([-0.4, -0.3, -0.2, 0.2, 0.3, 0.4])
    reference = rng.normal(size=64)
    diffs = rng.normal(size=(len(offsets), 64))

    observed = cosine(fit_axis_and_drift(offsets, diffs)[0], reference)
    null = permutation_cosines(offsets, diffs, reference, resamples=400, seed=5)

    assert permutation_p_value(observed, null, alternative="greater") > 0.05


def test_the_null_is_reproducible_from_its_seed() -> None:
    rng = np.random.default_rng(11)
    offsets = np.array([-0.3, -0.1, 0.1, 0.3])
    _, diffs = linear_family(rng, offsets)
    reference = rng.normal(size=64)

    first = permutation_cosines(offsets, diffs, reference, resamples=50, seed=9)
    assert np.array_equal(first, permutation_cosines(offsets, diffs, reference, resamples=50, seed=9))


def test_p_value_can_never_be_exactly_zero() -> None:
    """A finite number of resamples cannot license a p of 0."""
    null = np.zeros(100)
    assert permutation_p_value(-1.0, null, alternative="less") == pytest.approx(1 / 101)


def test_an_unknown_alternative_is_refused() -> None:
    with pytest.raises(ValueError, match="'less', 'greater' or 'two-sided'"):
        permutation_p_value(0.0, np.zeros(10), alternative="sideways")


def test_the_permutation_null_matches_the_least_squares_it_replaces() -> None:
    """The fast path is an algebraic rewrite, not an approximation.

    Permuting offsets leaves the diffs alone, so the axis's cosine against a
    reference can be had from the Gram matrix and one projection instead of a
    fresh least squares over the whole parameter vector each time. If that is
    ever rewritten again, this is what says it still computes the same thing.
    """
    rng = np.random.default_rng(0)
    offsets = np.array([-0.45, -0.3, -0.2, -0.05, 0.05, 0.2, 0.3, 0.45])
    diffs = rng.normal(size=(8, 400)) + offsets[:, None] * rng.normal(size=400)
    reference = rng.normal(size=400)

    fast = permutation_cosines(offsets, diffs, reference, resamples=200, seed=7)

    replay = np.random.default_rng(7)
    slow = np.array([cosine(fit_axis_and_drift(replay.permutation(offsets), diffs)[0], reference) for _ in range(200)])

    assert np.allclose(fast, slow, atol=1e-12)


def test_the_permutation_null_still_refuses_a_degenerate_grid() -> None:
    diffs = np.random.default_rng(1).normal(size=(4, 20))
    with pytest.raises(ValueError, match="same offset"):
        permutation_cosines(np.zeros(4), diffs, np.ones(20), resamples=5)


def test_the_norm_null_separates_a_real_axis_from_none() -> None:
    """|axis| against zero is meaningless; against a shuffled grid it is not.

    Least squares returns a nonzero slope through any cloud of diffs, and how
    large depends on the noise and the grid's leverage. Both are what differ
    between the things a ladder compares, which is exactly why the raw norm
    cannot be read down one.
    """
    rng = np.random.default_rng(0)
    offsets = np.array([m * s for m in (0.45, 0.44, 0.30, 0.20, 0.10) for s in (1, -1)])
    axis = rng.normal(size=200)

    signal = np.outer(offsets, axis) + 0.5 * rng.normal(size=(len(offsets), 200))
    noise = 0.5 * rng.normal(size=(len(offsets), 200)) + rng.normal(size=200)

    for diffs, expected in ((signal, True), (noise, False)):
        observed = float(np.linalg.norm(fit_axis_and_drift(offsets, diffs)[0]))
        null = permutation_norms(offsets, diffs, resamples=400, seed=1)
        significant = permutation_p_value(observed, null, alternative="greater") < 0.05
        assert significant is expected


def test_reliability_is_high_on_a_real_axis_and_nil_on_noise() -> None:
    rng = np.random.default_rng(3)
    offsets = np.array([m * s for m in (0.45, 0.44, 0.43, 0.30, 0.20, 0.10) for s in (1, -1)])
    axis = rng.normal(size=200)

    signal = np.outer(offsets, axis) + 0.5 * rng.normal(size=(len(offsets), 200))
    noise = 0.5 * rng.normal(size=(len(offsets), 200)) + rng.normal(size=200)

    assert split_half_reliability(offsets, signal, splits=80, seed=4) > 0.7
    assert abs(split_half_reliability(offsets, noise, splits=80, seed=4)) < 0.3


def test_reliability_needs_four_pairs_to_split() -> None:
    """A grid with fewer cannot make two halves that each fit a slope."""
    offsets = np.array([0.4, -0.4, 0.2, -0.2])
    diffs = np.random.default_rng(5).normal(size=(4, 30))

    assert np.isnan(split_half_reliability(offsets, diffs, splits=20, seed=0))


def test_reliability_splits_pairs_so_each_half_stays_balanced() -> None:
    """An unbalanced half leaks the common fine-tuning component into its axis.

    Two halves that both leaked it would agree about the leak, and the agreement
    would be reported as reliability. Here every diff carries a large shared
    offset and nothing else: balanced halves must find no reliable direction.
    """
    rng = np.random.default_rng(6)
    offsets = np.array([m * s for m in (0.45, 0.44, 0.43, 0.30, 0.20, 0.10) for s in (1, -1)])
    common = 50.0 * rng.normal(size=300)
    diffs = np.tile(common, (len(offsets), 1)) + 0.5 * rng.normal(size=(len(offsets), 300))

    assert abs(split_half_reliability(offsets, diffs, splits=80, seed=7)) < 0.3


def test_reliability_finds_its_pairs_when_the_offsets_are_inexact() -> None:
    """Offsets are differences of floats, and the base value decides how inexact.

    Colour 0's grid is values minus 1.0, so +0.45 arrives as 0.4500000000000002
    and -0.45 as -0.44999999999999996. Matching magnitudes against rounded keys
    found no pairs at all and returned nan for every rung of a ladder, while
    colour 1 -- base 0.5, arithmetic exact -- looked fine.
    """
    base = 1.0
    values = [round(base + m * s, 2) for m in (0.45, 0.44, 0.43, 0.30, 0.20, 0.10) for s in (1, -1)]
    offsets = np.array([v - base for v in values])
    assert any(abs(o) not in {0.45, 0.44, 0.43, 0.30, 0.20, 0.10} for o in np.abs(offsets)), "need inexact offsets"

    rng = np.random.default_rng(8)
    diffs = np.outer(offsets, rng.normal(size=200)) + 0.5 * rng.normal(size=(len(offsets), 200))

    assert split_half_reliability(offsets, diffs, splits=80, seed=9) > 0.7


# --- storing diffs at the width they came from -------------------------------
#
# Checkpoints are float32, so a sweep of a large model is held at that width and
# the arithmetic widens where it needs to rather than the storage widening
# everywhere. What these guard is that "where it needs to" is right: a promotion
# that copies the arms-by-parameters matrix defeats the point, and an accumulator
# left at float32 quietly costs precision in a number quoted to three places.


def test_gram_matrix_is_float64_however_the_diffs_are_stored() -> None:
    rng = np.random.default_rng(20)
    diffs = rng.normal(size=(5, 301)).astype(np.float32)

    for chunk in (1, 17, 301, 10_000):
        gram = gram_matrix(diffs, chunk=chunk)
        assert gram.dtype == np.float64
        assert np.allclose(gram, diffs.astype(np.float64) @ diffs.astype(np.float64).T)


def test_project_is_float64_however_the_diffs_are_stored() -> None:
    rng = np.random.default_rng(21)
    diffs = rng.normal(size=(5, 301)).astype(np.float32)
    vector = rng.normal(size=301).astype(np.float32)

    for chunk in (1, 17, 301, 10_000):
        projected = project(diffs, vector, chunk=chunk)
        assert projected.dtype == np.float64
        assert np.allclose(projected, diffs.astype(np.float64) @ vector.astype(np.float64))


def test_a_fit_does_not_widen_the_matrix_it_was_handed() -> None:
    """float64 in, float64 out; float32 in, float32 out -- and no copy either way."""
    rng = np.random.default_rng(22)
    offsets = np.array([-0.3, -0.1, 0.1, 0.3])
    _, diffs = linear_family(rng, offsets, noise=0.05)

    assert fit_axis(offsets, diffs).dtype == np.float64
    assert fit_axis_and_drift(offsets, diffs)[0].dtype == np.float64
    assert fit_axis(offsets, diffs.astype(np.float32)).dtype == np.float32
    assert fit_axis_and_drift(offsets, diffs.astype(np.float32))[0].dtype == np.float32


def test_integer_diffs_are_still_promoted() -> None:
    offsets = np.array([-2.0, -1.0, 1.0, 2.0])
    diffs = np.array([[-2, -4], [-1, -2], [1, 2], [2, 4]])

    assert fit_axis(offsets, diffs).dtype == np.float64


def test_the_narrower_storage_changes_no_reported_number() -> None:
    """The whole justification: same statistics, half the memory."""
    rng = np.random.default_rng(23)
    offsets = np.array([-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4])
    wide = (rng.normal(size=4096) + np.outer(offsets, rng.normal(size=4096))).astype(np.float32).astype(np.float64)
    narrow = wide.astype(np.float32)

    axis_wide = fit_axis_and_drift(offsets, wide)[0]
    axis_narrow = fit_axis_and_drift(offsets, narrow)[0]

    assert cosine(axis_wide, axis_narrow) == pytest.approx(1.0, abs=1e-9)
    assert split_half_reliability(offsets, narrow, splits=40) == pytest.approx(
        split_half_reliability(offsets, wide, splits=40), abs=1e-9
    )
    # Looser here, and the reason is the reference rather than the machinery: the
    # null is measured against each path's *own* fitted axis, and those differ at
    # float32 resolution. What is left is 1e-5 on a quantity reported to three
    # places, which is the claim this test is making.
    assert permutation_cosines(offsets, narrow, axis_narrow, resamples=50, seed=3).mean() == pytest.approx(
        permutation_cosines(offsets, wide, axis_wide, resamples=50, seed=3).mean(), abs=1e-4
    )
