"""Tests for level ground truth recovered from an observation.

The cross-check against the solver is the one that matters. Everything the
distance-field probe claims rests on these labels being right, and a label bug
does not crash — it produces a slightly worse probe, which reads as a finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis import geometry
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import UNREACHABLE, distance_field, walls_blocking_other_objectives


def sampled(n, size=11, seed=0, max_size=None):
    """Real levels and their encodings, which is what the analysis layer sees."""
    sampler = MazeLevelSampler(size_range=(size, size))
    encoder = ObservationEncoder(max_size=max_size or size, n_features=2)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        level = sampler.sample(rng)
        yield level, encoder.encode(level, level.agent_start)


def test_fields_match_the_solver_computed_from_the_level():
    """The observation-space twin must agree with the code we already trust.

    `solver.walls_blocking_other_objectives` works from a Level; `blocking_walls`
    reconstructs the same thing from channels. If they disagree, every label the
    distance probe is scored against is wrong.
    """
    checked = 0
    for level, observation in sampled(100):
        for index, objective in enumerate(level.objectives):
            feature = objective.feature_id
            ours = geometry.bfs_field(
                geometry.blocking_walls(observation, feature, n_features=2),
                geometry.objective_cell(observation, feature),
            )
            theirs = distance_field(walls_blocking_other_objectives(level, index), objective.position)

            height, width = level.shape
            reference = theirs.astype(np.float64)
            reference[theirs == UNREACHABLE] = np.nan
            np.testing.assert_array_equal(
                ours[:height, :width],
                reference,
                err_msg="observation-derived field disagrees with the solver",
            )
            checked += 1
    assert checked >= 200, f"expected two objectives per level, only compared {checked} fields"


def test_chebyshev_never_exceeds_manhattan_never_exceeds_bfs():
    """A free property that catches transposed indices and metric mix-ups.

    Manhattan is the free-space geodesic for 4-way moves, so walls can only ever
    make the true distance longer; Chebyshev ignores one axis entirely.
    """
    for _, observation in sampled(40):
        source = geometry.objective_cell(observation, 0)
        shape = observation.shape[:2]
        bfs = geometry.bfs_field(geometry.blocking_walls(observation, 0, n_features=2), source)
        manhattan = geometry.manhattan_field(shape, source)
        chebyshev = geometry.chebyshev_field(shape, source)

        known = np.isfinite(bfs)
        assert np.all(chebyshev <= manhattan), "Chebyshev exceeded Manhattan"
        assert np.all(manhattan[known] <= bfs[known]), "a maze route was shorter than the straight line"


def test_a_cell_behind_the_other_objective_is_unreachable():
    """Reaching an objective ends the episode, so a route through one never arrives.

    A corridor whose only path to feature 0 passes feature 1: finite without
    blocking, NaN with it.
    """
    observation = np.zeros((5, 5, 5), dtype=np.float32)
    observation[:, :, 0] = 1.0
    observation[2, 1:4, 0] = 0.0  # a one-cell corridor: agent, then f1, then f0
    observation[2, 1, 1] = 1.0
    observation[2, 3, 2] = 1.0  # feature 0 at the far end
    observation[2, 2, 3] = 1.0  # feature 1 in the middle

    open_field = geometry.bfs_field(observation[:, :, 0] > 0.5, geometry.objective_cell(observation, 0))
    blocked = geometry.bfs_field(
        geometry.blocking_walls(observation, 0, n_features=2), geometry.objective_cell(observation, 0)
    )

    agent = geometry.agent_cell(observation)
    assert open_field[agent] == 2, "the corridor is two steps long when nothing blocks it"
    assert np.isnan(blocked[agent]), "feature 1 sits between the agent and feature 0, so it cannot be reached"


def test_unreachable_cells_are_nan_and_never_minus_one():
    """The silent catastrophe this module exists to prevent.

    -1 regressed onto a field whose mean is 12 does not crash; it reads as a
    slightly worse probe.
    """
    seen_unreachable = False
    for _, observation in sampled(60):
        field = geometry.bfs_field(geometry.blocking_walls(observation, 0, n_features=2), geometry.agent_cell(observation))
        assert not np.any(field == UNREACHABLE), "an unreachable cell survived as the number -1"
        seen_unreachable |= bool(np.any(np.isnan(field)))
    assert seen_unreachable, "no unreachable cells in 60 levels — the test is not exercising the path"


def test_padding_does_not_move_the_field():
    """Levels smaller than the observation are padded with wall; distances inside
    the maze must be unaffected."""
    for level, observation in sampled(20, size=7, max_size=11):
        height, width = level.shape
        assert observation.shape[:2] == (11, 11)
        field = geometry.bfs_field(geometry.blocking_walls(observation, 0, n_features=2), geometry.agent_cell(observation))
        assert np.all(np.isnan(field[height:, :])), "padding outside the maze must be unreachable"
        assert np.isfinite(field[:height, :width]).any(), "the maze itself must still be reachable"


def test_check_layout_rejects_a_wrong_n_features():
    """Five channels is 2+2+1 here but 2+1+2 under a broadcast encoder, so a
    wrong n_features would silently read the value channel as an objective."""
    _, observation = next(iter(sampled(1)))
    geometry.check_layout(observation, n_features=2)

    # Counting marked cells is not enough to catch this: the value channel holds
    # 1.0 at one objective and 0.5 at the other, so at a 0.5 threshold it also
    # marks exactly one cell. Only "is this a 0/1 indicator" separates them.
    with pytest.raises(ValueError, match="not a 0/1 indicator"):
        geometry.check_layout(observation, n_features=3)


def test_agent_and_objective_cells_are_distinct_and_free():
    for _, observation in sampled(30):
        free = geometry.free_cells(observation)
        cells = [geometry.agent_cell(observation)] + [geometry.objective_cell(observation, f) for f in (0, 1)]
        assert all(free[cell] for cell in cells), "the agent or an objective was placed inside a wall"
        assert len(set(cells)) == 3, "the agent started on an objective"
