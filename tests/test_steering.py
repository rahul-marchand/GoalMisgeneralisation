"""Tests for calibrated steering directions.

The calibration test is the one that matters. Steering is only interpretable if
"add alpha" really moves the decoded quantity by alpha — otherwise the slope
this whole experiment reports is measured against the wrong x-axis, and it will
look like a weak causal effect rather than a broken conversion.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from goalmisgen.analysis import steering
from goalmisgen.analysis.probes import apply_linear, fit_ridge


@dataclasses.dataclass
class FakeLayer:
    """Stands in for cleanba's LSTMCellState, which is a flax PyTreeNode."""

    c: np.ndarray
    h: np.ndarray

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


def fitted_probe(depth=12, n=800, seed=0):
    """A probe whose target is a known linear function of the activations."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, depth)) * rng.uniform(0.5, 4.0, depth)
    truth = rng.normal(size=depth)
    y = x @ truth + 7.0
    return x, y, fit_ridge(x, y, l2=1e-8)


def test_steering_by_alpha_moves_the_decoded_value_by_alpha():
    """The claim the whole intervention rests on."""
    x, _, (weights, mean, std) = fitted_probe()
    direction = steering.from_probe("d", weights, std)

    for alpha in (-6.0, -1.0, 0.5, 3.0, 9.0):
        before = apply_linear(x, weights, mean, std)
        after = apply_linear(x + direction.scaled(alpha), weights, mean, std)
        np.testing.assert_allclose(after - before, alpha, atol=1e-6)
        assert steering.verify(direction, weights, mean, std, alpha) == pytest.approx(alpha, abs=1e-6)


def test_the_direction_is_the_smallest_change_that_does_it():
    """Any other shift achieving the same decoded move must be at least as big.

    A larger-than-necessary perturbation disturbs more of the network than the
    quantity being steered, which is exactly what the control directions exist
    to detect — so the real one must be minimal.
    """
    _, _, (weights, mean, std) = fitted_probe()
    direction = steering.from_probe("d", weights, std)
    effective = weights[:-1] / std

    rng = np.random.default_rng(1)
    for _ in range(20):
        other = rng.normal(size=len(effective))
        achieved = float(effective @ other)
        if abs(achieved) < 1e-6:
            continue
        equivalent = other / achieved  # rescaled to move the decoded value by 1
        assert np.linalg.norm(direction.delta) <= np.linalg.norm(equivalent) + 1e-9


def test_standardisation_is_undone_not_ignored():
    """Channels have very different scales here. A direction that forgot to
    divide by std would decode to something proportional but wrong, which reads
    as a slope far from 1 rather than as an error."""
    _, _, (weights, mean, std) = fitted_probe()
    assert std.max() / std.min() > 3, "the fixture must have unequal channel scales to test this"

    naive = weights[:-1] / (weights[:-1] @ weights[:-1])  # the mistake: no std
    achieved = float((weights[:-1] / std) @ naive)
    assert abs(achieved - 1.0) > 0.2, "the fixture cannot distinguish the mistake from the correct direction"


def test_applying_to_a_carry_splits_across_layers_in_order():
    delta = np.arange(6, dtype=np.float64)
    carry = [
        FakeLayer(c=np.zeros((2, 3, 3, 2)), h=np.zeros((2, 3, 3, 2))),
        FakeLayer(c=np.zeros((2, 3, 3, 4)), h=np.zeros((2, 3, 3, 4))),
    ]
    steered = steering.apply_to_carry(carry, delta)

    np.testing.assert_allclose(steered[0].h[0, 0, 0], [0.0, 1.0])
    np.testing.assert_allclose(steered[1].h[0, 0, 0], [2.0, 3.0, 4.0, 5.0])
    assert all(np.all(layer.c == 0) for layer in steered), "the cell state must be left alone"
    # Every cell gets the same shift: the field moves uniformly, not locally.
    np.testing.assert_allclose(steered[1].h[1, 2, 2], [2.0, 3.0, 4.0, 5.0])


def test_a_direction_that_does_not_fill_the_carry_is_an_error():
    """Silently steering only the first layers would look like a weak effect."""
    carry = [FakeLayer(c=np.zeros((1, 2, 2, 3)), h=np.zeros((1, 2, 2, 3)))]
    with pytest.raises(ValueError, match="consumed|needs at least"):
        steering.apply_to_carry(carry, np.zeros(5))


def test_control_directions_match_the_real_one_in_size():
    """A control must differ in where it points and nothing else."""
    _, _, (weights, mean, std) = fitted_probe()
    real = steering.from_probe("real", weights, std)

    random = steering.matched_random("random", real)
    assert random.unit_norm == pytest.approx(real.unit_norm)

    _, _, (other_weights, _, other_std) = fitted_probe(seed=5)
    other = steering.matched("other", steering.from_probe("o", other_weights, other_std), real)
    assert other.unit_norm == pytest.approx(real.unit_norm)


def test_a_random_direction_barely_moves_the_decoded_value():
    """The control's premise: a same-sized shift in a random direction should
    not move the quantity being steered."""
    _, _, (weights, mean, std) = fitted_probe(depth=96)
    real = steering.from_probe("real", weights, std)
    random = steering.matched_random("random", real, seed=3)

    intended = steering.verify(real, weights, mean, std, 6.0)
    accidental = steering.verify(random, weights, mean, std, 6.0)
    assert intended == pytest.approx(6.0, abs=1e-6)
    assert abs(accidental) < 0.25 * abs(intended), f"the control moved the decoded value by {accidental:.2f}"


def test_a_contrast_direction_is_calibrated_like_a_probe_one():
    """Difference of means points where the network travels; the probe is used
    only to scale it, so one unit still moves the decoded value by one."""
    x, y, (weights, mean, std) = fitted_probe(depth=40, n=2000)
    high, low = x[y > np.quantile(y, 0.75)], x[y < np.quantile(y, 0.25)]

    contrast = steering.from_contrast("contrast", high, low, weights, std)
    for alpha in (-4.0, 1.0, 7.0):
        assert steering.verify(contrast, weights, mean, std, alpha) == pytest.approx(alpha, abs=1e-6)


def test_the_contrast_direction_differs_from_the_probe_direction():
    """If they coincided there would be nothing to gain by switching. Ridge
    minimises norm; a contrast follows the data."""
    x, y, (weights, mean, std) = fitted_probe(depth=40, n=2000)
    high, low = x[y > np.quantile(y, 0.75)], x[y < np.quantile(y, 0.25)]

    probe = steering.from_probe("probe", weights, std)
    contrast = steering.from_contrast("contrast", high, low, weights, std)

    cosine = float(probe.delta @ contrast.delta / (probe.unit_norm * contrast.unit_norm))
    assert abs(cosine) < 0.99, f"the two directions are nearly identical (cosine {cosine:.3f})"
    assert probe.unit_norm <= contrast.unit_norm + 1e-9, "the probe direction must be the smaller of the two"


def test_a_contrast_that_does_not_separate_the_quantity_is_rejected():
    """Calibrating by a near-zero shift would scale the direction to nonsense."""
    _, _, (weights, _, std) = fitted_probe(depth=10)
    same = np.ones((5, 10))
    with pytest.raises(ValueError, match="cannot be calibrated"):
        steering.from_contrast("flat", same, same, weights, std)


def test_layer_slicing_takes_the_requested_third():
    from goalmisgen.analysis.probes import Feature, layer_slice

    grid = np.arange(2 * 2 * 6, dtype=np.float64).reshape(2, 2, 6)
    whole = Feature("whole", lambda rollout: grid)

    first = layer_slice(whole, 0, 3)(None)
    last = layer_slice(whole, 2, 3)(None)
    np.testing.assert_array_equal(first, grid[..., 0:2])
    np.testing.assert_array_equal(last, grid[..., 4:6])
    assert layer_slice(whole, 0, 3).name == "whole[layer 0]"


def test_layer_slicing_refuses_an_uneven_split():
    from goalmisgen.analysis.probes import Feature, layer_slice

    whole = Feature("whole", lambda rollout: np.zeros((2, 2, 7)))
    with pytest.raises(ValueError, match="do not divide"):
        layer_slice(whole, 0, 3)(None)
