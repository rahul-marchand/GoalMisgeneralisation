"""Tests for the probing machinery.

The AUC implementation and the wall-masking both silently distort results if
wrong — a probe that scores 0.9 because walls are trivially negative looks like
a finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.probes import cell_dataset, probe, probe_by_distance, roc_auc


class FakeRollout:
    def __init__(self, features, observation, visited, visit_step):
        self.features, self.observation = features, observation
        self.visited, self.visit_step = visited, visit_step


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

        features = rng.normal(size=(size, size, 8)).astype(np.float32)
        if informative:
            marked = visited if marked_until is None else (visited & (visit_step <= marked_until))
            features[marked, 0] += 4.0  # one channel marks the route
        out.append(FakeRollout(features, obs, visited, visit_step))
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
    last = max(b.step for b in full)
    assert next(b for b in full if b.step == last).auc > 0.95, "a planted route must decode at its far end"
    assert next(b for b in near if b.step == 0).auc > 0.95, "the near end is marked in both"
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
