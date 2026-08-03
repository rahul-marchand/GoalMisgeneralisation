"""Guards that the task actually requires trading value off against distance.

The optimal choice maximises ``value - step_penalty * distance``. If the step
penalty is too small, "go to the highest value" is always right and distance
never has to be estimated. If it is too large, "go to the nearest" is always
right and value never has to be read. Either extreme deletes the comparison
mechanism the project exists to study, and neither would be caught by any
correctness test — the environment would be perfectly well-behaved and
scientifically useless.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.configs.env import MazeConfig
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import objective_distances
from goalmisgen.envs.values import FixedValues, UniformValues

N_LEVELS = 600
MAX_HEURISTIC_ACCURACY = 90.0
"""Neither single-cue heuristic may exceed this against the true optimum."""


def heuristic_accuracies(sampler: MazeLevelSampler, step_penalty: float) -> tuple[float, float]:
    """Accuracy of "pick the highest value" and "pick the nearest", as percentages."""
    rng = np.random.default_rng(0)
    values, distances = [], []
    for _ in range(N_LEVELS):
        level = sampler.sample(rng)
        level_distances = objective_distances(level)
        if any(distance is None for distance in level_distances):
            continue
        values.append([objective.value for objective in level.objectives])
        distances.append(level_distances)

    value_array = np.array(values)
    distance_array = np.array(distances, dtype=float)

    utilities = value_array - step_penalty * distance_array
    optimal = utilities.argmax(axis=1)

    by_value = value_array.argmax(axis=1)
    by_distance = distance_array.argmin(axis=1)
    return 100 * float((by_value == optimal).mean()), 100 * float((by_distance == optimal).mean())


def test_default_configuration_needs_both_cues():
    config = MazeConfig(max_episode_steps=120)
    by_value, by_distance = heuristic_accuracies(config.sampler(), config.step_penalty)

    assert by_value < MAX_HEURISTIC_ACCURACY, (
        f"'go to the highest value' alone is {by_value:.1f}% correct: the step penalty is too "
        "small and distance is irrelevant"
    )
    assert by_distance < MAX_HEURISTIC_ACCURACY, (
        f"'go to the nearest' alone is {by_distance:.1f}% correct: the step penalty is too " "large and value is irrelevant"
    )


@pytest.mark.parametrize("randomise", [False, True])
def test_both_value_schemes_stay_in_the_balanced_regime(randomise):
    config = MazeConfig(max_episode_steps=120, randomise_values=randomise)
    by_value, by_distance = heuristic_accuracies(config.sampler(), config.step_penalty)
    assert by_value < MAX_HEURISTIC_ACCURACY
    assert by_distance < MAX_HEURISTIC_ACCURACY


def test_a_tiny_penalty_is_detected_as_degenerate():
    """Confirm the guard above is not passing vacuously."""
    sampler = MazeLevelSampler(size_range=(9, 15), values=FixedValues((1.0, 0.5)))
    by_value, _ = heuristic_accuracies(sampler, step_penalty=0.001)
    assert by_value > 95.0


def test_a_huge_penalty_is_detected_as_degenerate():
    sampler = MazeLevelSampler(size_range=(9, 15), values=FixedValues((1.0, 0.5)))
    _, by_distance = heuristic_accuracies(sampler, step_penalty=5.0)
    assert by_distance > 95.0


def test_randomised_values_populate_the_decision_boundary():
    """Near-ties are where the comparison mechanism is doing real work.

    Randomising values varies the value gap episode to episode, which spreads
    levels across the decision boundary instead of clustering them away from it.
    """

    def near_tie_fraction(sampler, step_penalty: float) -> float:
        rng = np.random.default_rng(0)
        margins = []
        for _ in range(N_LEVELS):
            level = sampler.sample(rng)
            distances = objective_distances(level)
            if any(distance is None for distance in distances):
                continue
            utilities = [objective.value - step_penalty * distance for objective, distance in zip(level.objectives, distances)]
            margins.append(abs(utilities[0] - utilities[1]))
        return float((np.array(margins) < 0.05).mean())

    fixed = MazeLevelSampler(size_range=(9, 15), values=FixedValues((1.0, 0.5)))
    random_values = MazeLevelSampler(size_range=(9, 15), values=UniformValues(0.25, 1.0))

    assert near_tie_fraction(random_values, 0.03) > near_tie_fraction(fixed, 0.03)
