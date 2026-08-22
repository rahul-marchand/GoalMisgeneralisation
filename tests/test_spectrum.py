"""Tests for counting the directions a family of diffs spans.

The registered prediction is that the second direction *does not* replicate, so
the failure that matters is a module which reports agreement where there is
none. Most of these tests build a family with a known rank and check the answer
comes back; the load-bearing ones build noise and check it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.spectrum import (
    axis_removed_operator,
    drift_removed_operator,
    gram_matrix,
    participation_ratio,
    permutation_participation_ratio,
    residual_reliability,
    spectrum,
    variance_shares,
)
from goalmisgen.analysis.weights import fit_axis_and_drift

OFFSETS = np.array([-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4])


def rank_one(rng: np.random.Generator, offsets: np.ndarray, width: int = 128, noise: float = 0.0):
    """One axis and one drift, exactly as the one-knob account describes."""
    axis, drift = rng.normal(size=width), rng.normal(size=width)
    diffs = drift + np.outer(offsets, axis) + noise * rng.normal(size=(len(offsets), width))
    return axis, drift, diffs


def test_gram_matrix_matches_the_direct_product_across_chunk_sizes() -> None:
    rng = np.random.default_rng(0)
    diffs = rng.normal(size=(6, 97))
    for chunk in (1, 7, 97, 1000):
        assert np.allclose(gram_matrix(diffs, chunk=chunk), diffs @ diffs.T)


def test_operators_reproduce_the_least_squares_fit() -> None:
    """The whole module rests on drift and axis being linear in the arms."""
    rng = np.random.default_rng(1)
    _, _, diffs = rank_one(rng, OFFSETS, noise=0.3)
    axis, drift = fit_axis_and_drift(OFFSETS, diffs)

    assert np.allclose(drift_removed_operator(OFFSETS) @ diffs, diffs - drift)
    assert np.allclose(axis_removed_operator(OFFSETS) @ diffs, diffs - drift - np.outer(OFFSETS, axis))


def test_a_clean_line_has_one_direction() -> None:
    rng = np.random.default_rng(2)
    _, _, diffs = rank_one(rng, OFFSETS)
    operator = drift_removed_operator(OFFSETS)

    values = spectrum(operator @ gram_matrix(diffs) @ operator.T)

    assert participation_ratio(values) == pytest.approx(1.0, abs=1e-6)
    assert variance_shares(values)[0] == pytest.approx(1.0, abs=1e-6)


def test_two_planted_directions_raise_the_participation_ratio() -> None:
    rng = np.random.default_rng(3)
    axis, drift, _ = rank_one(rng, OFFSETS)
    second = rng.normal(size=128)
    second -= second @ axis / (axis @ axis) * axis
    second *= np.linalg.norm(axis) / np.linalg.norm(second)
    # Matched in energy as well as in length: the participation ratio counts
    # directions by how much they carry, so a second direction loaded a quarter
    # as hard is two thirds of a direction, not one.
    loading = np.abs(OFFSETS) - np.abs(OFFSETS).mean()
    loading *= np.linalg.norm(OFFSETS) / np.linalg.norm(loading)
    diffs = drift + np.outer(OFFSETS, axis) + np.outer(loading, second)
    operator = drift_removed_operator(OFFSETS)

    ratio = participation_ratio(spectrum(operator @ gram_matrix(diffs) @ operator.T))

    assert ratio == pytest.approx(2.0, abs=0.1)


def test_a_planted_second_direction_replicates() -> None:
    """A real second axis is the outcome that would redirect the campaign, so it must be findable."""
    rng = np.random.default_rng(4)
    offsets = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4])
    axis, drift, _ = rank_one(rng, offsets)
    second = rng.normal(size=128)
    diffs = (
        drift
        + np.outer(offsets, axis)
        + np.outer(np.abs(offsets) - np.abs(offsets).mean(), second)
        + 0.02 * rng.normal(size=(len(offsets), 128))
    )

    assert residual_reliability(offsets, gram_matrix(diffs), splits=60, seed=0) > 0.7


def test_noise_alone_does_not_replicate() -> None:
    """The test that protects the registered prediction: no structure, no agreement."""
    rng = np.random.default_rng(5)
    offsets = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4])
    axis, drift, _ = rank_one(rng, offsets)
    diffs = drift + np.outer(offsets, axis) + 0.5 * rng.normal(size=(len(offsets), 128))

    assert residual_reliability(offsets, gram_matrix(diffs), splits=60, seed=0) < 0.3


def test_replication_is_blind_to_the_sign_of_a_singular_vector() -> None:
    """Flipping a half's leading vector is an eigensolver's choice, not a disagreement."""
    rng = np.random.default_rng(6)
    offsets = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4])
    axis, drift, _ = rank_one(rng, offsets)
    second = rng.normal(size=128)
    diffs = drift + np.outer(offsets, axis) + np.outer(np.abs(offsets) - np.abs(offsets).mean(), second)

    assert residual_reliability(offsets, gram_matrix(diffs), splits=40, seed=0) > 0.9


def test_too_few_mirrored_pairs_gives_nan_rather_than_a_number() -> None:
    rng = np.random.default_rng(7)
    offsets = np.array([-0.2, -0.1, 0.1, 0.2, 0.35])
    _, _, diffs = rank_one(rng, offsets, noise=0.1)

    assert np.isnan(residual_reliability(offsets, gram_matrix(diffs), splits=20))


def test_the_permutation_null_is_not_assumed_to_be_one() -> None:
    """Removing a line fitted to noise leaves structure; the null has to say how much."""
    rng = np.random.default_rng(8)
    _, _, diffs = rank_one(rng, OFFSETS, noise=0.4)

    drawn = permutation_participation_ratio(OFFSETS, gram_matrix(diffs), resamples=50, seed=0)

    assert len(drawn) == 50
    assert np.all(drawn >= 1.0)


def test_shapes_and_degenerate_families_are_refused_by_name() -> None:
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="at least three arms"):
        drift_removed_operator(np.array([-0.1, 0.1]))
    with pytest.raises(ValueError, match="same offset"):
        drift_removed_operator(np.array([0.2, 0.2, 0.2]))
    with pytest.raises(ValueError, match="square"):
        spectrum(rng.normal(size=(3, 4)))
    with pytest.raises(ValueError, match="no length"):
        participation_ratio(np.zeros(4))
    with pytest.raises(ValueError, match="offsets against"):
        residual_reliability(OFFSETS, np.eye(3))
