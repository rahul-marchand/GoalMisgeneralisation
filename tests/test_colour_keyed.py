"""Colours pinned to objectives, for sweeps whose values cross parity.

The scheme exists because of a measurement that looked like a result: arms
trained past the point where the swept objective overtakes the other produced
thresholds that turned around and climbed again. Nothing was broken -- the level
generator paints colour 0 on whichever objective is worth most, so those arms
were learning a *relabelled* preference, and the threshold was tracking the
absolute value gap. These tests pin the distinction down.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.configs.env import MazeConfig
from goalmisgen.envs.colour_keyed import ColourKeyedFeatures
from goalmisgen.envs.features import CorrelatedFeatures
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.values import FixedValues


def test_the_correlated_scheme_repaints_when_the_swept_objective_overtakes() -> None:
    """The behaviour being worked around, asserted so it cannot surprise twice."""
    rng = np.random.default_rng(0)
    assert CorrelatedFeatures(1.0).assign((1.0, 0.5), rng) == (0, 1)
    assert CorrelatedFeatures(1.0).assign((1.0, 1.4), rng) == (1, 0)


def test_colours_stay_on_their_objectives_across_parity() -> None:
    rng = np.random.default_rng(0)
    scheme = ColourKeyedFeatures()
    for values in ((1.0, 0.5), (1.0, 1.0), (1.0, 1.4), (0.3, 0.5)):
        assert scheme.assign(values, rng) == (0, 1), values


def test_generation_is_unchanged_so_stored_levels_stay_valid() -> None:
    """``canonical`` is what generation uses; content must not depend on this."""
    assert ColourKeyedFeatures().canonical() == CorrelatedFeatures(1.0)


def test_the_config_selects_the_scheme() -> None:
    default = MazeConfig(max_episode_steps=120)
    keyed = MazeConfig(max_episode_steps=120, colour_keyed_features=True)
    assert isinstance(default.feature_scheme(), CorrelatedFeatures)
    assert isinstance(keyed.feature_scheme(), ColourKeyedFeatures)


def test_a_sampled_crossing_level_keeps_colour_zero_on_objective_zero() -> None:
    """The end-to-end version: a level whose colour 1 is the richer objective."""
    sampler = MazeLevelSampler(
        size_range=(11, 11),
        values=FixedValues((1.0, 1.4)),
        features=ColourKeyedFeatures(),
    )
    level = sampler.sample(np.random.default_rng(3))
    by_colour = {objective.feature_id: objective.value for objective in level.objectives}
    assert by_colour[0] == 1.0 and by_colour[1] == 1.4

    correlated = MazeLevelSampler(
        size_range=(11, 11),
        values=FixedValues((1.0, 1.4)),
        features=CorrelatedFeatures(1.0),
    )
    swapped = correlated.sample(np.random.default_rng(3))
    by_colour = {objective.feature_id: objective.value for objective in swapped.objectives}
    assert by_colour[0] == 1.4, "the correlated scheme should paint the richer objective colour 0"


def test_demonstrations_can_pin_colours_too(tmp_path) -> None:
    """The BC stream reads colours from its demonstrations, not from an env.

    A crossing-value demo set built with the correlated scheme would put colour
    0 on the objective that overtook, so the route model would be taught the
    same relabelled preference the env arms were, one layer further back.
    """
    import numpy as np

    from goalmisgen.envs.dataset import LevelDataset
    from goalmisgen.envs.sampling import MazeLevelSampler
    from goalmisgen.envs.values import FixedValues
    from goalmisgen.offline import demos as demos_module

    del tmp_path
    sampler = MazeLevelSampler(size_range=(11, 11), values=FixedValues((1.0, 1.4)))
    dataset = LevelDataset.generate(sampler, n_levels=8, seed=0, block_size=8)

    keyed = demos_module.demonstrate_block(
        dataset, np.arange(len(dataset)), rho=1.0, seed=0,
        step_penalty=0.05, step_limit=120, max_actions=64, colour_keyed=True,
    )
    correlated = demos_module.demonstrate_block(
        dataset, np.arange(len(dataset)), rho=1.0, seed=0,
        step_penalty=0.05, step_limit=120, max_actions=64,
    )
    richest = np.argmax(keyed["values"], axis=1)
    rows = np.arange(len(dataset))
    assert (keyed["feature_ids"][rows, 0] == 0).all(), "objective 0 should keep colour 0"
    assert (correlated["feature_ids"][rows, richest] == 0).all(), "the correlated scheme repaints"
