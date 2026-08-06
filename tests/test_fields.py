"""Tests for the distance-field probe.

The straight-line control is the one that matters. It is the direct successor to
`test_a_pure_distance_feature_does_not_survive_distance_matching`: the same
class of confound, in regression form, on the same rig that failed to catch it
the first time.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from goalmisgen.analysis import fields
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import MazeLevelSampler


def make_rollouts(n, size=11, seed=0):
    """Real levels, so the geometry the confound exploits is the real geometry."""
    sampler = MazeLevelSampler(size_range=(size, size))
    encoder = ObservationEncoder(max_size=size, n_features=2)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        level = sampler.sample(rng)
        observation = encoder.encode(level, level.agent_start)
        out.append(types.SimpleNamespace(observation=observation, features=None))
    return out


def truth_grid(rollout):
    """A feature holding exactly the field being asked for."""
    labels, _, _ = fields.distance_target(rollout, 0)
    return np.nan_to_num(labels)[:, :, None]


def straight_grid(rollout):
    """A feature holding only free-space geometry — no maze solving in it."""
    _, manhattan, chebyshev = fields.distance_target(rollout, 0)
    return np.stack([manhattan, chebyshev], axis=-1)


def noise_grid(rollout, depth=96, seed=0):
    shape = rollout.observation.shape[:2]
    return np.random.default_rng(seed).normal(size=(*shape, depth))


def datasets(grid, n_train=40, n_test=25):
    return (
        fields.field_dataset(make_rollouts(n_train, seed=0), grid, feature_id=0),
        fields.field_dataset(make_rollouts(n_test, seed=1), grid, feature_id=0),
    )


def test_a_straight_line_feature_does_not_survive_the_detour_test():
    """The control that decides whether this measurement means anything.

    A feature holding only Manhattan and Chebyshev distance contains no maze
    solving whatever, yet it explains about a third of the variance in the true
    field, because corr(bfs, manhattan) = 0.57 on these levels. Pooled R² would
    read that as a finding. The detour subset and the stratified correlation
    must both refuse it.
    """
    result = fields.field_probe("straight-line", *datasets(straight_grid))

    assert result.pooled_r2 > 0.20, (
        f"pooled R² was only {result.pooled_r2:.3f} — if the confound is this weak the test is not "
        "exercising the trap it exists to document"
    )
    assert result.detour_r2 < 0.10, f"straight-line geometry still explained detour cells: {result.detour_r2:.3f}"
    assert abs(result.partial_r) < 0.20, f"straight-line geometry survived stratification: {result.partial_r:.3f}"


def test_the_rig_detects_a_field_that_is_present():
    """Without this a null result is uninterpretable — it could just mean the
    pipeline is broken."""
    result = fields.field_probe("bfs-only", *datasets(truth_grid))
    assert result.detour_r2 > 0.95, f"the rig failed to find a field it was handed outright: {result.detour_r2:.3f}"
    assert result.partial_r > 0.9, f"a perfect feature did not survive stratification: {result.partial_r:.3f}"


def test_noise_features_score_at_chance():
    result = fields.field_probe("noise", *datasets(noise_grid))
    assert result.detour_r2 < 0.10, f"noise explained detour cells: {result.detour_r2:.3f}"
    assert abs(result.partial_r) < 0.20, f"noise survived stratification: {result.partial_r:.3f}"


def test_dimensionality_does_not_decide_the_winner():
    """A 1-dimensional perfect feature must not lose to a 96-dimensional one
    just because a single fixed penalty suits one and not the other."""

    def padded(rollout):
        return np.concatenate([truth_grid(rollout), noise_grid(rollout, depth=95)], axis=-1)

    lean = fields.field_probe("truth", *datasets(truth_grid))
    fat = fields.field_probe("truth+noise", *datasets(padded))

    assert lean.detour_r2 > 0.95, f"the 1-d arm lost to regularisation: {lean.detour_r2:.3f}"
    assert fat.detour_r2 > 0.80, f"the same signal padded with noise collapsed: {fat.detour_r2:.3f}"


def test_stratified_correlation_of_a_pure_function_of_the_stratum_is_zero():
    """The property the metric exists for: anything the confound already
    determines contributes nothing after centring within it."""
    stratum = np.repeat(np.arange(20), 8)
    prediction = stratum * 2.5
    y = stratum * -1.5 + 3.0
    assert abs(fields.stratified_correlation(prediction, y, stratum)) < 1e-9


def test_r2_is_measured_against_the_rows_it_is_given():
    """A subset with a larger mean must not be credited for that."""
    y = np.array([10.0, 12.0, 14.0, 16.0])
    assert fields.r2(y, y) == pytest.approx(1.0)
    assert fields.r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)
    # Predicting the *global* mean of a wider population would look good here if
    # r2 used that population's mean rather than these rows'.
    assert fields.r2(y, np.full_like(y, 0.0)) < 0.0


def test_the_objective_cell_is_dropped():
    """Distance zero, and a one-hot channel in the observation — every arm would
    score it. Same trap as the route probe's step 0."""
    rollouts = make_rollouts(5, seed=3)
    data = fields.field_dataset(rollouts, truth_grid, feature_id=0)
    assert data.y.min() > 0, "the objective's own cell survived the mask"


def test_unreachable_cells_never_enter_the_dataset():
    """A leaked -1 in a field whose mean is 12 reads as a slightly worse probe."""
    data = fields.field_dataset(make_rollouts(30, seed=4), truth_grid, feature_id=0)
    assert np.isfinite(data.y).all()
    assert data.y.min() >= 1
    assert data.mask_fraction < 1.0, "no cells were masked — the unreachable path is untested"


def test_uncertainty_is_estimated_over_episodes_not_cells():
    """Cells in an episode share one maze and one field. Resampling them
    independently would report a far larger experiment than this is."""
    episode = np.repeat(np.arange(12), 30)
    values = np.repeat(np.random.default_rng(0).normal(size=12), 30)

    low, high = fields.bootstrap_episodes(lambda rows: float(values[rows].mean()), episode, resamples=200)
    per_cell = values.std() / np.sqrt(len(values))
    assert (high - low) > 4 * per_cell, "the interval is as narrow as if cells were independent"


def test_stratified_correlation_needs_comparable_rows():
    """One member per stratum means nothing can be compared within it. That is
    missing power, not a zero result, so it must not report as one."""
    stratum = np.arange(20)
    prediction = np.random.default_rng(0).normal(size=20)
    assert np.isnan(fields.stratified_correlation(prediction, prediction, stratum))
