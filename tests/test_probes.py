"""Tests for the probing machinery.

The AUC implementation and the wall-masking both silently distort results if
wrong — a probe that scores 0.9 because walls are trivially negative looks like
a finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.probes import (
    apply_linear,
    apply_multinomial,
    auc_interval,
    cell_dataset,
    class_directions,
    class_write_accuracy,
    fit_multinomial,
    fit_ridge,
    probe,
    probe_by_distance,
    roc_auc,
)
from goalmisgen.envs.solver import distance_field


class FakeRollout:
    def __init__(self, features, observation, visited, visit_step, distance):
        self.features, self.observation = features, observation
        self.visited, self.visit_step, self.distance = visited, visit_step, distance


def make_rollouts(n, size=7, informative=True, seed=0, marked_until=None):
    """Rollouts whose features encode the route (or don't).

    ``marked_until`` marks only the first few steps of the route, imitating a
    probe that has found an imminent action rather than a plan.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        obs = np.zeros((size, size, 5), dtype=np.float32)
        obs[0, :, 0] = obs[-1, :, 0] = obs[:, 0, 0] = obs[:, -1, 0] = 1.0  # wall border
        visited = np.zeros((size, size), dtype=bool)
        visited[size // 2, 1:-1] = True  # a corridor route
        visit_step = np.full((size, size), -1, dtype=np.int16)
        visit_step[size // 2, 1:-1] = np.arange(size - 2)
        distance = distance_field(obs[:, :, 0] > 0.5, (size // 2, 1))

        features = rng.normal(size=(size, size, 8)).astype(np.float32)
        if informative:
            marked = visited if marked_until is None else (visited & (visit_step <= marked_until))
            features[marked, 0] += 4.0  # one channel marks the route
        out.append(FakeRollout(features, obs, visited, visit_step, distance))
    return out


def test_auc_matches_hand_computed_values():
    assert roc_auc(np.array([0, 0, 1, 1.0]), np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)
    assert roc_auc(np.array([1, 1, 0, 0.0]), np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(0.0)
    assert roc_auc(np.array([0, 1, 0, 1.0]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_walls_are_excluded_so_they_cannot_inflate_scores():
    rollouts = make_rollouts(4, size=7)
    x, y = cell_dataset(rollouts, mask_walls=True)
    free_cells_per_level = 25  # 7x7 with a one-cell border
    assert len(y) == 4 * free_cells_per_level

    x_all, y_all = cell_dataset(rollouts, mask_walls=False)
    assert len(y_all) == 4 * 49
    assert y_all.mean() < y.mean(), "walls are trivially negative and dilute the positive rate"


def test_distance_split_separates_a_plan_from_an_imminent_action():
    """The check the whole distance breakdown exists to make.

    Both agents score well overall, so only the split can tell them apart: one
    has the full route in its features, the other only its next two steps.
    """
    full = probe_by_distance(make_rollouts(40, seed=0), make_rollouts(20, seed=1))
    near = probe_by_distance(
        make_rollouts(40, seed=0, marked_until=1), make_rollouts(20, seed=1, marked_until=1)
    )

    assert {b.step for b in full} == {b.step for b in near}
    # Step 0 is absent by construction: the agent's own cell is the only cell at
    # distance 0, so it has no matched negatives. It was trivially AUC 1.000 for
    # every feature set, the raw observation included, and proved nothing.
    assert 0 not in {b.step for b in full}

    first, last = min(b.step for b in full), max(b.step for b in full)
    assert next(b for b in full if b.step == last).auc > 0.95, "a planted route must decode at its far end"
    assert next(b for b in near if b.step == first).auc > 0.95, "the near end is marked in both"
    assert next(b for b in near if b.step == last).auc < 0.7, "an unmarked far end must not decode"


def test_probe_recovers_a_route_that_is_present():
    train, test = make_rollouts(30, seed=0), make_rollouts(15, seed=1)
    result = probe(train, test)
    assert result.auc > 0.95, f"failed to find a route encoded in the features: {result}"


def test_probe_finds_nothing_when_the_route_is_absent():
    """Guards against a probe that scores well on structure alone."""
    train = make_rollouts(30, informative=False, seed=0)
    test = make_rollouts(15, informative=False, seed=1)
    result = probe(train, test)
    assert result.auc < 0.65, f"probe found signal in noise: {result}"


@pytest.mark.slow
def test_collect_rollouts_runs_against_a_real_vector_env():
    """Exercises the capture path end to end.

    The rollout loop touches gymnasium's autoreset semantics and the object
    array it returns as final_info, neither of which any unit test reaches — the
    first version of this code raised on `final_info or [...]` and only failed
    when run for real.
    """
    import jax

    from goalmisgen.analysis import collect_rollouts
    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.configs.presets import maze_drc33

    config = MazeConfig(max_episode_steps=20, num_envs=8, min_size=5, max_size=5, asynchronous=False)
    envs = config.make()
    args = maze_drc33(min_size=5, max_size=5)
    policy, _, params = args.net.init_params(envs, jax.random.PRNGKey(0))

    rollouts = collect_rollouts(envs, policy, params, n_episodes=8, seed=0)
    assert len(rollouts) == 8
    for r in rollouts:
        assert r.features.shape[:2] == (5, 5)
        assert r.features.shape[2] == 3 * 32, "three layers of 32 channels each"
        assert r.visited.any(), "the agent must occupy at least its start cell"
        assert r.visited.sum() <= 25
        assert r.visit_step[r.visited].min() == 0, "the start cell is reached at step 0"
        assert (r.visit_step[~r.visited] == -1).all(), "unvisited cells carry no arrival step"

        # The route must be a contiguous run of arrival steps. Autoreset hands
        # back the *next* level on the terminating step, so reading it drops the
        # objective - the deepest cell of the route - and adds a disconnected
        # cell belonging to a different maze.
        steps = np.sort(r.visit_step[r.visited])
        assert np.array_equal(steps, np.arange(len(steps))), f"route has gaps or duplicates: {steps}"
        if r.info.get("reached_objective"):
            assert steps[-1] == r.info["episode_steps"], "the objective cell must be the last one recorded"


def test_a_pure_distance_feature_does_not_survive_distance_matching():
    """The control that invalidated the first version of this measurement.

    A feature holding nothing but distance-from-agent contains no plan, yet it
    scored 0.99 down to 0.56 across the bands when negatives were pooled over
    all distances — the same "still high far out" shape that was read as
    evidence of planning. Matched negatives must reduce it to chance.
    """

    def distance_only(n, seed):
        rollouts = make_rollouts(n, size=9, informative=False, seed=seed)
        for r in rollouts:
            reachable = r.distance >= 0
            r.features = np.zeros((*r.visited.shape, 1), dtype=np.float64)
            r.features[reachable, 0] = -r.distance[reachable]
        return rollouts

    bands = probe_by_distance(distance_only(40, 0), distance_only(20, 1))
    assert bands, "the control must produce comparable bands at all"
    worst = max(abs(b.auc - 0.5) for b in bands)
    assert worst < 0.15, f"distance alone still decodes after matching: {[(b.step, round(b.auc, 3)) for b in bands]}"


def test_uncertainty_is_estimated_over_episodes_not_cells():
    """Cells within an episode share a maze and a route.

    Resampling cells would treat a few hundred episodes as a few thousand
    independent samples and report an interval far too narrow to be honest.
    """
    train, test = make_rollouts(30, seed=0), make_rollouts(20, seed=1)
    low, high = auc_interval(train, test, resamples=100)

    result = probe(train, test)
    assert low <= result.auc <= high, "the point estimate must lie inside its own interval"
    assert high - low > 0, "a bootstrap over 20 episodes cannot be exactly certain"

    # Identical rollouts differ only by noise, so a wider sample must not widen
    # the interval without bound - this catches resampling the wrong axis.
    assert high - low < 0.5


def test_ridge_recovers_a_known_linear_map():
    """With a weak penalty and no noise, the fit must return the map it was given."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    truth = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
    y = x @ truth + 7.0

    w, mean, std = fit_ridge(x, y, l2=1e-8)
    np.testing.assert_allclose(apply_linear(x, w, mean, std), y, atol=1e-6)


def test_ridge_leaves_the_target_in_its_own_units():
    """Predictions are read as distances in cells, so a fit that standardised the
    target would silently rescale every error. Rescaling the target must rescale
    the predictions with it."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = 10.0 + x[:, 0] * 4.0

    plain = apply_linear(x, *fit_ridge(x, y, l2=1e-8))
    scaled = apply_linear(x, *fit_ridge(x, 100.0 * y + 5.0, l2=1e-8))

    np.testing.assert_allclose(plain, y, atol=1e-6)
    np.testing.assert_allclose(scaled, 100.0 * y + 5.0, rtol=1e-6)


def test_ridge_does_not_shrink_the_intercept():
    """A heavy penalty must flatten the slope toward zero while the prediction
    still sits at the target's mean, not at zero."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 3))
    y = 25.0 + x[:, 0]

    w, mean, std = fit_ridge(x, y, l2=1e6)
    prediction = apply_linear(x, w, mean, std)
    assert abs(prediction.mean() - 25.0) < 0.1, "the intercept was penalised"
    assert prediction.std() < 0.05, "a heavy penalty should have flattened the slope"


def test_a_written_class_actually_wins_the_argmax():
    """The intervention's claim, checked. Raising class k's logit is not the same
    as making k the prediction: a rival whose weight vector correlates with k's
    and is longer wins at every magnitude, so "add w_k" can be a no-op no scale
    rescues. The direction here is the one that maximises the margin over all
    rivals, and this asserts it does."""
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.2, 5.0, 8)
    centres = rng.normal(size=(4, 8)) * 3
    y = rng.integers(0, 4, 2000)
    x = (centres[y] + rng.normal(size=(2000, 8))) * scale

    weights, mean, std = fit_multinomial(x, y, 4)
    assert (apply_multinomial(x, weights, mean, std).argmax(1) == y).mean() > 0.9

    directions, margins = class_directions(weights, std)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert (margins > 0).all()
    assert class_write_accuracy(x, weights, mean, std, directions, magnitude=30.0) == 1.0


def test_the_reported_margin_is_the_margin_actually_achieved():
    """An approximate solver reported positive margins for directions whose true
    margin against one rival was negative — a direction that does nothing, with a
    number beside it saying otherwise. The reported value has to be checkable
    against the thing it describes."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 3000)
    x = rng.normal(size=(3000, 10)) * rng.uniform(0.2, 5.0, 10)
    x += np.eye(5, 10)[y] * 4

    weights, mean, std = fit_multinomial(x, y, 5)
    effective = (weights[:-1] / std[:, None]).T
    directions, margins = class_directions(weights, std)

    for index in range(5):
        rivals = effective[index] - np.delete(effective, index, axis=0)
        assert (rivals @ directions[index]).min() == pytest.approx(margins[index], abs=1e-9)


def test_an_unwritable_class_raises():
    """A class whose weight vector sits inside the cone of the others cannot be
    written by any additive edit. Silently returning a direction for it would
    make a null result look like the network ignoring the intervention."""
    # Weights are (depth + 1, n_classes): three dimensions, a bias row, four
    # classes. The fourth class sits exactly at the centroid of the other three,
    # so no direction raises it above all of them.
    weights = np.zeros((4, 4))
    weights[:3, :3] = np.eye(3)
    weights[:3, 3] = weights[:3, :3].mean(axis=1)
    with pytest.raises(ValueError, match="cannot be written"):
        class_directions(weights, np.ones(3))


def test_multinomial_does_not_shrink_a_rare_class_out_of_existence():
    """The bias row is unpenalised for the same reason ridge's intercept is: a
    NEVER class holding 80% of cells would otherwise swallow the four directions."""
    rng = np.random.default_rng(1)
    y = np.concatenate([np.zeros(1900), np.ones(100)]).astype(int)
    x = np.concatenate([rng.normal(size=(1900, 4)), rng.normal(size=(100, 4)) + 6])

    weights, mean, std = fit_multinomial(x, y, 2)
    predicted = apply_multinomial(x, weights, mean, std).argmax(1)
    assert predicted[y == 1].mean() > 0.9
