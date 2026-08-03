"""Tests for level sampling.

The distributional tests here guard the project's central experimental
variable. If the realised value/colour correlation does not match the requested
one, every misgeneralisation result downstream is measuring the wrong thing, so
these are checked statistically rather than by smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.generation import RecursiveBacktracker
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import solve
from goalmisgen.envs.values import FixedValues, UniformValues

N_TRIALS = 4000
"""Enough that a 0.03 tolerance is ~4 standard errors. Seeded, so deterministic."""

SMALL_MAZE = (5, 5)
"""Feature assignment is independent of layout, so the correlation tests use the
smallest legal maze and stay fast despite drawing thousands of samples."""


def feature_zero_marks_the_best(sampler: MazeLevelSampler, n_trials: int = N_TRIALS, seed: int = 0) -> float:
    """Fraction of levels in which feature 0 sits on the highest-value objective."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_trials):
        level = sampler.sample(rng)
        values = [objective.value for objective in level.objectives]
        best = int(np.argmax(values))
        hits += level.objectives[best].feature_id == 0
    return hits / n_trials


# --------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------


def test_sampled_levels_are_valid_and_solvable():
    sampler = MazeLevelSampler()
    rng = np.random.default_rng(0)
    for _ in range(200):
        level = sampler.sample(rng)
        # Level validates positions on construction; solve() additionally proves
        # every objective is reachable from the agent.
        solution = solve(level, step_penalty=0.01)
        assert all(distance is not None for distance in solution.distances)


def test_maze_sizes_stay_within_range_and_stay_odd():
    sampler = MazeLevelSampler(size_range=(5, 15))
    rng = np.random.default_rng(0)
    sizes = {sampler.sample(rng).shape[0] for _ in range(300)}
    assert sizes <= {5, 7, 9, 11, 13, 15}
    assert len(sizes) > 1, "size should actually vary"


def test_mazes_are_square():
    sampler = MazeLevelSampler()
    level = sampler.sample(np.random.default_rng(0))
    assert level.shape[0] == level.shape[1]


@pytest.mark.parametrize("n_objectives", [1, 2, 3, 4])
def test_feature_ids_are_a_permutation(n_objectives):
    sampler = MazeLevelSampler(
        n_objectives=n_objectives,
        values=FixedValues(tuple(1.0 - 0.1 * i for i in range(n_objectives))),
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        level = sampler.sample(rng)
        feature_ids = sorted(objective.feature_id for objective in level.objectives)
        assert feature_ids == list(range(n_objectives))


def test_sampling_is_deterministic_given_a_seed():
    sampler = MazeLevelSampler()
    first = sampler.sample(np.random.default_rng(11))
    second = sampler.sample(np.random.default_rng(11))

    assert np.array_equal(first.walls, second.walls)
    assert first.agent_start == second.agent_start
    assert first.objectives == second.objectives


def test_uniform_values_produce_varying_values():
    sampler = MazeLevelSampler(values=UniformValues(), feature_value_correlation=1.0)
    rng = np.random.default_rng(0)
    observed = {tuple(objective.value for objective in sampler.sample(rng).objectives) for _ in range(20)}
    assert len(observed) == 20


# --------------------------------------------------------------------------
# The correlation knob
# --------------------------------------------------------------------------


def test_perfect_correlation_always_marks_the_best_objective():
    sampler = MazeLevelSampler(size_range=SMALL_MAZE, feature_value_correlation=1.0)
    assert feature_zero_marks_the_best(sampler, n_trials=500) == 1.0


def test_zero_correlation_never_marks_the_best_objective():
    """rho=0 is perfect anti-correlation, the sharpest possible test shift."""
    sampler = MazeLevelSampler(size_range=SMALL_MAZE, feature_value_correlation=0.0)
    assert feature_zero_marks_the_best(sampler, n_trials=500) == 0.0


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_realised_correlation_matches_the_requested_one(rho):
    sampler = MazeLevelSampler(size_range=SMALL_MAZE, feature_value_correlation=rho)
    assert feature_zero_marks_the_best(sampler) == pytest.approx(rho, abs=0.03)


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_correlation_holds_with_more_than_two_objectives(rho):
    sampler = MazeLevelSampler(
        size_range=SMALL_MAZE,
        n_objectives=3,
        values=FixedValues((1.0, 0.6, 0.3)),
        feature_value_correlation=rho,
    )
    assert feature_zero_marks_the_best(sampler) == pytest.approx(rho, abs=0.03)


def test_correlation_holds_with_randomised_values():
    """With UniformValues the best objective is not at a fixed index."""
    sampler = MazeLevelSampler(size_range=SMALL_MAZE, values=UniformValues(), feature_value_correlation=0.75)
    assert feature_zero_marks_the_best(sampler) == pytest.approx(0.75, abs=0.03)


def test_single_objective_ignores_correlation():
    sampler = MazeLevelSampler(n_objectives=1, values=FixedValues((1.0,)), feature_value_correlation=0.3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        level = sampler.sample(rng)
        assert level.objectives[0].feature_id == 0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_rejects_out_of_range_correlation():
    with pytest.raises(ValueError, match="feature_value_correlation"):
        MazeLevelSampler(feature_value_correlation=1.5)


def test_rejects_zero_objectives():
    with pytest.raises(ValueError, match="n_objectives"):
        MazeLevelSampler(n_objectives=0)


def test_rejects_even_maze_sizes():
    with pytest.raises(ValueError, match="odd"):
        MazeLevelSampler(size_range=(4, 10))


def test_rejects_an_empty_size_range():
    with pytest.raises(ValueError, match="size_range is empty"):
        MazeLevelSampler(size_range=(15, 5))


def test_rejects_a_maze_too_small_for_the_objectives():
    sampler = MazeLevelSampler(
        size_range=(3, 3),
        n_objectives=4,
        values=FixedValues((1.0, 0.8, 0.6, 0.4)),
    )
    with pytest.raises(ValueError, match="free cells"):
        sampler.sample(np.random.default_rng(0))


def test_accepts_a_custom_generator():
    sampler = MazeLevelSampler(generator=RecursiveBacktracker(braid_probability=0.5))
    level = sampler.sample(np.random.default_rng(0))
    solution = solve(level, step_penalty=0.01)
    assert all(distance is not None for distance in solution.distances)
