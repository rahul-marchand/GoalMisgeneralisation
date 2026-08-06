"""Tests for the per-cell regression probe.

The null-arm test is the one that matters. It is the direct successor to
`test_a_pure_distance_feature_does_not_survive_distance_matching`: the same
class of confound, in regression form, on the rig that failed to catch it the
first time.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from goalmisgen.analysis import fields, targets
from goalmisgen.analysis.probes import Feature
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import MazeLevelSampler

FIXED = targets.DistanceToObjective(select=targets.fixed(0), name="d->f0")


def make_rollouts(n, size=11, seed=0):
    """Real levels, so the geometry the confound exploits is the real geometry."""
    sampler = MazeLevelSampler(size_range=(size, size))
    encoder = ObservationEncoder(max_size=size, n_features=2)
    rng = np.random.default_rng(seed)
    return [
        types.SimpleNamespace(
            observation=encoder.encode(level, level.agent_start),
            info={"reached_feature_id": 0},
            index=index,
        )
        for index, level in ((i, sampler.sample(rng)) for i in range(n))
    ]


def noise_feature(depth=96, seed=0):
    def grid(rollout):
        return np.random.default_rng(seed + rollout.index).normal(size=(*rollout.observation.shape[:2], depth))

    return Feature("noise", grid)


def datasets(build, n_train=40, n_test=25, target=FIXED):
    """``build`` maps a rollout list to the Feature read off it."""
    train_rollouts = make_rollouts(n_train, seed=0)
    test_rollouts = make_rollouts(n_test, seed=1)
    return (
        fields.cell_data(train_rollouts, build(train_rollouts), target),
        fields.cell_data(test_rollouts, build(test_rollouts), target),
    )


def oracle(rollouts):
    return targets.controls(FIXED, rollouts)[0]


def null(rollouts):
    return targets.controls(FIXED, rollouts)[1]


def shuffled(rollouts):
    return targets.controls(FIXED, rollouts)[2]


def test_the_null_arm_does_not_survive_the_hard_cell_test():
    """The control that decides whether this measurement means anything.

    Straight-line geometry contains no maze solving, yet it explains about a
    third of the variance in the true field because corr(bfs, manhattan) = 0.57
    on these levels. Pooled R² reads that as a finding. The hard subset and the
    stratified correlation must both refuse it.
    """
    result = fields.field_probe("null", *datasets(null))

    assert result.pooled_r2 > 0.15, (
        f"pooled R² was only {result.pooled_r2:.3f} — if the confound is this weak the test is not "
        "exercising the trap it documents"
    )
    assert result.hard_r2 < 0.10, f"straight-line geometry explained the hard cells: {result.hard_r2:.3f}"
    assert abs(result.partial_r) < 0.20, f"straight-line geometry survived stratification: {result.partial_r:.3f}"


def test_the_rig_detects_a_field_that_is_present():
    """Without this a null result cannot be told from a broken pipeline."""
    result = fields.field_probe("oracle", *datasets(oracle))
    assert result.hard_r2 > 0.95, f"the rig failed to find a field handed to it: {result.hard_r2:.3f}"
    assert result.partial_r > 0.9


def test_a_field_on_the_wrong_maze_scores_at_chance():
    """The oracle arm cannot catch a consistent feature/label misalignment — it
    would be misaligned identically and still score 1.000. This can."""
    result = fields.field_probe("shuffled", *datasets(shuffled))
    assert result.hard_r2 < 0.10, f"a field from another maze explained this one: {result.hard_r2:.3f}"
    assert abs(result.partial_r) < 0.20


def test_noise_scores_at_chance():
    result = fields.field_probe("noise", *datasets(lambda rollouts: noise_feature()))
    assert result.hard_r2 < 0.10
    assert abs(result.partial_r) < 0.20


def test_shrinkage_is_removed_before_scoring_and_still_reported():
    """A deliberately compressed feature orders perfectly but is scaled wrong.
    The pilot could not tell that apart from a weak representation."""

    def squashed(rollouts):
        del rollouts
        return Feature("squashed", lambda r: (0.25 * np.nan_to_num(FIXED.labels(r)))[:, :, None])

    result = fields.field_probe("squashed", *datasets(squashed))

    assert result.hard_shape_r2 > 0.95, "the ordering is perfect and must be reported as such"
    assert result.hard_r2 > 0.9, "after recalibration a perfectly-ordered feature must score well"


def test_degenerate_columns_are_dropped_and_counted():
    """Standardising divides by std + 1e-8, so a constant channel becomes
    unit-scale noise the penalty must suppress — weakening exactly the arms with
    least signal, which are the controls."""

    def padded(rollouts):
        del rollouts

        def grid(rollout):
            labels = np.nan_to_num(FIXED.labels(rollout))[:, :, None]
            return np.concatenate([labels, np.zeros((*labels.shape[:2], 8))], axis=-1)

        return Feature("padded", grid)

    result = fields.field_probe("padded", *datasets(padded))
    assert result.dropped_columns == 8, f"constant channels survived: {result.dropped_columns}"
    assert result.depth == 1
    assert result.hard_r2 > 0.95, "dropping dead channels must not cost the real one"


def test_a_clipped_penalty_is_flagged():
    """A winner at the end of the grid was clipped, not chosen. The pilot's
    observation arm selected the grid maximum and nobody noticed."""
    train, _ = datasets(null)
    _, interior = fields.choose_l2(train, grid=(1e-8, 1e-7))
    assert not interior, "a two-point grid can only be clipped, and must say so"


def test_a_confound_that_exceeds_its_labels_is_rejected():
    """Column 0 must lower-bound the label or the hard subset is nonsense."""

    class Broken:
        name = "broken"
        confound_names = ("too-big",)

        def labels(self, rollout):
            return FIXED.labels(rollout)

        def confound(self, rollout):
            return (np.nan_to_num(FIXED.labels(rollout)) + 5.0)[:, :, None]

    rollouts = make_rollouts(5)
    with pytest.raises(ValueError, match="lower bound"):
        fields.cell_data(rollouts, oracle(rollouts), Broken())


def test_unscoreable_cells_never_enter_the_dataset():
    rollouts = make_rollouts(30, seed=4)
    data = fields.cell_data(rollouts, oracle(rollouts), FIXED)
    assert np.isfinite(data.y).all()
    assert data.y.min() >= 1, "the objective's own cell survived"
    assert data.mask_fraction < 1.0, "nothing was masked — the unreachable path is untested"


def test_an_all_masked_target_is_an_error_not_an_empty_table():
    """A target that silently yields no rows would print a table of NaN."""

    class Empty:
        name = "empty"
        confound_names = ("none",)

        def labels(self, rollout):
            return np.full(rollout.observation.shape[:2], np.nan)

        def confound(self, rollout):
            return np.full((*rollout.observation.shape[:2], 1), np.nan)

    rollouts = make_rollouts(3)
    with pytest.raises(ValueError, match="no scoreable cells"):
        fields.cell_data(rollouts, oracle(rollouts), Empty())
