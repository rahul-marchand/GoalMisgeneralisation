"""Tests for probe targets and the controls they generate.

The confound requirement is the point of this module: a target that cannot say
what would beat it for free is a target whose result cannot be read. These tests
enforce that, and check the two invariants the generic scoring layer relies on —
labels are NaN wherever a cell must not be scored, and confound column 0 is a
lower bound on the label.
"""

from __future__ import annotations

import functools
import types

import numpy as np
import pytest

from goalmisgen.analysis import geometry, targets
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import MazeLevelSampler


def make_rollouts(n, size=11, seed=0, reached_feature=0):
    """Real levels, with the outcome fields the selectors read."""
    sampler = MazeLevelSampler(size_range=(size, size))
    encoder = ObservationEncoder(max_size=size, n_features=2)
    rng = np.random.default_rng(seed)
    out = []
    for index in range(n):
        level = sampler.sample(rng)
        info = {} if reached_feature is None else {"reached_feature_id": reached_feature}
        out.append(
            types.SimpleNamespace(
                observation=encoder.encode(level, level.agent_start),
                features=rng.normal(size=(size, size, 8)),
                info=dict(info),
                index=index,
            )
        )
    return out


FIXED = targets.DistanceToObjective(select=targets.fixed(0), name="d->f0")
REACHED = targets.DistanceToObjective(select=targets.reached, name="d->reached")
UNREACHED = targets.DistanceToObjective(select=functools.partial(targets.unreached, n_features=2), name="d->unreached")


def test_a_fixed_target_reproduces_the_field_the_pilot_measured():
    """The keying is new; the field itself must not have changed. Compared
    against geometry directly, which the pilot also called."""
    for rollout in make_rollouts(30):
        expected = geometry.bfs_field(
            geometry.blocking_walls(rollout.observation, 0, 2), geometry.objective_cell(rollout.observation, 0)
        )
        expected[expected == 0] = np.nan  # the objective's own cell, dropped by both

        actual = FIXED.labels(rollout)
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_array_equal(actual[np.isfinite(actual)], expected[np.isfinite(expected)])


def test_reached_and_unreached_disagree_and_follow_the_outcome():
    for feature in (0, 1):
        rollout = make_rollouts(1, reached_feature=feature)[0]
        assert targets.reached(rollout) == feature
        assert targets.unreached(rollout) == 1 - feature

        reached_field = REACHED.labels(rollout)
        unreached_field = UNREACHED.labels(rollout)
        finite = np.isfinite(reached_field) & np.isfinite(unreached_field)
        assert finite.sum() > 5, "the two fields should overlap on most cells"
        assert not np.array_equal(reached_field[finite], unreached_field[finite]), "the two objectives gave one field"


def test_a_timed_out_episode_contributes_nothing():
    """No objective was reached, so 'distance to the one it chose' is undefined.
    Contributing the level anyway would silently key it to feature 0."""
    rollout = make_rollouts(1, reached_feature=None)[0]
    assert targets.reached(rollout) is None
    assert np.isnan(REACHED.labels(rollout)).all()
    assert np.isnan(REACHED.confound(rollout)).all()


def test_unreached_refuses_more_than_two_objectives():
    """With three objectives there is no single 'the other one'."""
    with pytest.raises(ValueError, match="only defined for two"):
        targets.unreached(make_rollouts(1)[0], n_features=3)


def test_every_target_declares_a_confound_that_lower_bounds_its_labels():
    """The generic layer picks out cells where the null is most wrong by
    subtracting column 0 from the label. If that can go negative the subset is
    nonsense, and the metric silently scores the wrong cells."""
    for target in (FIXED, REACHED, UNREACHED):
        assert target.confound_names, f"{target.name} declared no confound"
        for rollout in make_rollouts(40, seed=7):
            labels = target.labels(rollout)
            confound = target.confound(rollout)
            assert confound.shape[:2] == labels.shape
            assert confound.shape[2] == len(target.confound_names)

            known = np.isfinite(labels)
            assert np.all(confound[known, 0] <= labels[known] + 1e-9), (
                f"{target.name}: straight-line distance exceeded the true distance, "
                "so it is not a lower bound and the detour subset is meaningless"
            )


def test_labels_are_nan_on_walls_and_at_the_objective_itself():
    for rollout in make_rollouts(20, seed=2):
        labels = FIXED.labels(rollout)
        assert np.isnan(labels[~geometry.free_cells(rollout.observation)]).all(), "a wall was scoreable"
        assert np.nanmin(labels) >= 1, "the objective's own cell survived, and every arm would score it"


def test_controls_are_generated_from_the_target():
    """Not written out beside it — that is what stops the next question being
    asked without them."""
    rollouts = make_rollouts(12, seed=5)
    names = [feature.name for feature in targets.controls(FIXED, rollouts)]
    assert names == ["oracle:d->f0", "null:manhattan+chebyshev", "shuffled:d->f0"]


def test_the_shuffled_oracle_never_lands_on_its_own_episode():
    """A grid that stayed put would score 1.000 and quietly pass, which is the
    exact failure this control exists to catch."""
    rollouts = make_rollouts(25, seed=6)
    _, _, shuffled = targets.controls(FIXED, rollouts)

    matched = 0
    for rollout in rollouts:
        own = np.nan_to_num(FIXED.labels(rollout))[:, :, None]
        matched += int(np.array_equal(shuffled(rollout), own))
    assert matched == 0, f"{matched} shuffled grids were still attached to their own maze"


def test_the_shuffled_oracle_still_holds_real_fields():
    """It must be a real field on the wrong maze, not noise — otherwise it tests
    nothing the noise arm does not already test."""
    rollouts = make_rollouts(15, seed=8)
    _, _, shuffled = targets.controls(FIXED, rollouts)
    values = np.concatenate([shuffled(rollout).ravel() for rollout in rollouts])
    assert values.max() > 5, "the permuted grids do not look like distance fields"


def value_and_distance_rollouts(n, size=11, seed=0):
    """Real levels with the outcome recorded, for the ground-truth selectors."""
    return make_rollouts(n, size=size, seed=seed)


def test_the_selectors_agree_with_the_solver():
    """richer, nearer and best_utility must name the same objectives the
    environment's own ground truth does, or every split is mislabelled."""
    from goalmisgen.envs.observation import ObservationEncoder
    from goalmisgen.envs.sampling import MazeLevelSampler
    from goalmisgen.envs.solver import solve

    sampler = MazeLevelSampler(size_range=(11, 11))
    encoder = ObservationEncoder(max_size=11, n_features=2)
    rng = np.random.default_rng(3)

    checked = 0
    for _ in range(60):
        level = sampler.sample(rng)
        rollout = types.SimpleNamespace(observation=encoder.encode(level, level.agent_start), info={}, index=0)
        solution = solve(level, targets.STEP_PENALTY)
        if solution.is_ambiguous:
            continue

        feature_of = {index: objective.feature_id for index, objective in enumerate(level.objectives)}
        assert targets.best_utility(rollout) == feature_of[solution.optimal_index], "utility selector disagrees with solve()"

        values = [objective.value for objective in level.objectives]
        assert targets.richer(rollout) == feature_of[int(np.argmax(values))]

        distances = [d for d in solution.distances]
        if len(set(distances)) == len(distances):
            assert targets.nearer(rollout) == feature_of[int(np.argmin(distances))]
        checked += 1
    assert checked > 30, f"only {checked} unambiguous levels — the test is not exercising much"


def test_a_tie_is_dropped_rather_than_broken():
    """An objective pair indistinguishable on a criterion would be noise on both
    sides of the comparison it is supposed to split."""
    rollout = make_rollouts(1)[0]
    observation = rollout.observation
    value = geometry.value_channel(2)
    row0, col0 = geometry.objective_cell(observation, 0)
    row1, col1 = geometry.objective_cell(observation, 1)
    observation[row0, col0, value] = observation[row1, col1, value] = 0.75

    assert targets.richer(rollout) is None, "a value tie should drop the episode"


def test_the_coin_flip_control_is_deterministic_per_episode():
    """A re-run must produce the same split, or two runs are not comparable."""
    rollouts = make_rollouts(20, seed=11)
    once = [targets.coinflip(rollout) for rollout in rollouts]
    assert once == [targets.coinflip(rollout) for rollout in rollouts]
    assert 0 < sum(once) < len(once), "the control split put every episode on one side"
