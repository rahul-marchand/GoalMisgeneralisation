"""Tests for observation encoding.

The most important test here is that reordering ``Level.objectives`` leaves the
observation unchanged. If objective index leaked into the channel layout, the
agent could read value off the ordering and would never need the colour cue,
which would silently invalidate every misgeneralisation result.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.level import Level, Objective
from goalmisgen.envs.observation import (
    AGENT_CHANNEL,
    FIRST_FEATURE_CHANNEL,
    WALL_CHANNEL,
    ObservationEncoder,
)

OPEN_ROOM = np.ones((5, 5), dtype=np.bool_)
OPEN_ROOM[1:4, 1:4] = False


def make_level(objectives=None) -> Level:
    if objectives is None:
        objectives = (
            Objective((1, 3), value=1.0, feature_id=0),
            Objective((3, 1), value=0.5, feature_id=1),
        )
    return Level(walls=OPEN_ROOM, agent_start=(1, 1), objectives=objectives)


def test_shape_and_channel_count():
    encoder = ObservationEncoder(max_size=25, n_features=2, value_encoding="at_objective")
    assert encoder.n_channels == 5  # wall, agent, 2 features, 1 value
    assert encoder.shape == (25, 25, 5)


@pytest.mark.parametrize(
    "encoding,expected",
    [("none", 4), ("at_objective", 5), ("broadcast", 6)],
)
def test_channel_count_per_value_encoding(encoding, expected):
    encoder = ObservationEncoder(max_size=9, n_features=2, value_encoding=encoding)
    assert encoder.n_channels == expected


def test_observation_is_float32_and_hwc():
    encoder = ObservationEncoder(max_size=9)
    observation = encoder.encode(make_level(), (1, 1))
    assert observation.dtype == np.float32
    assert observation.shape == (9, 9, encoder.n_channels)


def test_padding_is_wall():
    encoder = ObservationEncoder(max_size=9)
    observation = encoder.encode(make_level(), (1, 1))

    # The 5x5 level sits top-left; everything beyond it is wall.
    assert observation[5:, :, WALL_CHANNEL].all()
    assert observation[:, 5:, WALL_CHANNEL].all()
    # And the level's own walls are preserved.
    assert np.array_equal(observation[:5, :5, WALL_CHANNEL], OPEN_ROOM.astype(np.float32))


def test_agent_channel_is_one_hot_at_the_given_position():
    encoder = ObservationEncoder(max_size=9)
    observation = encoder.encode(make_level(), (2, 2))
    assert observation[2, 2, AGENT_CHANNEL] == 1.0
    assert observation[:, :, AGENT_CHANNEL].sum() == 1.0


def test_agent_channel_tracks_movement_not_the_start():
    """The agent moves during an episode; encode takes its current position."""
    encoder = ObservationEncoder(max_size=9)
    level = make_level()
    moved = encoder.encode(level, (3, 3))
    assert moved[3, 3, AGENT_CHANNEL] == 1.0
    assert moved[level.agent_start[0], level.agent_start[1], AGENT_CHANNEL] == 0.0


def test_feature_channels_mark_objective_positions():
    encoder = ObservationEncoder(max_size=9)
    observation = encoder.encode(make_level(), (1, 1))
    assert observation[1, 3, FIRST_FEATURE_CHANNEL + 0] == 1.0
    assert observation[3, 1, FIRST_FEATURE_CHANNEL + 1] == 1.0
    assert observation[:, :, FIRST_FEATURE_CHANNEL + 0].sum() == 1.0
    assert observation[:, :, FIRST_FEATURE_CHANNEL + 1].sum() == 1.0


def test_objective_order_does_not_affect_the_observation():
    """Channels are keyed by feature, so index must be invisible to the agent."""
    encoder = ObservationEncoder(max_size=9)
    first = Objective((1, 3), value=1.0, feature_id=0)
    second = Objective((3, 1), value=0.5, feature_id=1)

    forwards = encoder.encode(make_level((first, second)), (1, 1))
    backwards = encoder.encode(make_level((second, first)), (1, 1))
    assert np.array_equal(forwards, backwards)


def test_swapping_which_feature_is_valuable_changes_the_observation():
    """Sanity check the previous test is not passing vacuously."""
    encoder = ObservationEncoder(max_size=9)
    normal = make_level(
        (
            Objective((1, 3), value=1.0, feature_id=0),
            Objective((3, 1), value=0.5, feature_id=1),
        )
    )
    swapped = make_level(
        (
            Objective((1, 3), value=0.5, feature_id=0),
            Objective((3, 1), value=1.0, feature_id=1),
        )
    )
    assert not np.array_equal(encoder.encode(normal, (1, 1)), encoder.encode(swapped, (1, 1)))


def test_at_objective_encoding_places_value_on_the_objective_cell():
    encoder = ObservationEncoder(max_size=9, value_encoding="at_objective")
    observation = encoder.encode(make_level(), (1, 1))
    channel = encoder.first_value_channel
    assert observation[1, 3, channel] == pytest.approx(1.0)
    assert observation[3, 1, channel] == pytest.approx(0.5)
    assert observation[:, :, channel].sum() == pytest.approx(1.5)


def test_broadcast_encoding_fills_a_plane_per_feature():
    encoder = ObservationEncoder(max_size=9, value_encoding="broadcast")
    observation = encoder.encode(make_level(), (1, 1))
    first = encoder.first_value_channel
    assert np.allclose(observation[:, :, first + 0], 1.0)
    assert np.allclose(observation[:, :, first + 1], 0.5)


def test_none_encoding_omits_value_channels():
    encoder = ObservationEncoder(max_size=9, n_features=2, value_encoding="none")
    assert encoder.n_value_channels == 0
    observation = encoder.encode(make_level(), (1, 1))
    assert observation.shape[2] == 4


def test_value_is_readable_independently_of_feature():
    """The agent must be able to tell value from something other than colour.

    Two levels with identical colours but swapped values must look different,
    otherwise a colour-following policy is the only strategy available and
    'misgeneralisation' would be forced rather than learned.
    """
    encoder = ObservationEncoder(max_size=9, value_encoding="at_objective")
    high_on_feature_zero = make_level(
        (
            Objective((1, 3), value=1.0, feature_id=0),
            Objective((3, 1), value=0.5, feature_id=1),
        )
    )
    high_on_feature_one = make_level(
        (
            Objective((1, 3), value=0.5, feature_id=0),
            Objective((3, 1), value=1.0, feature_id=1),
        )
    )
    a = encoder.encode(high_on_feature_zero, (1, 1))
    b = encoder.encode(high_on_feature_one, (1, 1))

    # Colour channels are identical...
    assert np.array_equal(
        a[:, :, FIRST_FEATURE_CHANNEL : FIRST_FEATURE_CHANNEL + 2],
        b[:, :, FIRST_FEATURE_CHANNEL : FIRST_FEATURE_CHANNEL + 2],
    )
    # ...but the observations differ, so value is perceivable on its own.
    assert not np.array_equal(a, b)


def test_encoded_observation_lies_in_the_declared_space():
    encoder = ObservationEncoder(max_size=9, value_encoding="broadcast")
    observation = encoder.encode(make_level(), (1, 1))
    assert encoder.space().contains(observation)


def test_rejects_a_level_larger_than_max_size():
    encoder = ObservationEncoder(max_size=3)
    with pytest.raises(ValueError, match="does not fit"):
        encoder.encode(make_level(), (1, 1))


def test_rejects_a_feature_id_beyond_n_features():
    encoder = ObservationEncoder(max_size=9, n_features=1)
    level = make_level()
    with pytest.raises(ValueError, match="n_features"):
        encoder.encode(level, (1, 1))


def test_rejects_a_value_outside_the_declared_range():
    encoder = ObservationEncoder(max_size=9, value_range=(0.0, 0.75))
    with pytest.raises(ValueError, match="value_range"):
        encoder.encode(make_level(), (1, 1))


def test_rejects_an_agent_position_in_a_wall():
    encoder = ObservationEncoder(max_size=9)
    with pytest.raises(ValueError, match="free cell"):
        encoder.encode(make_level(), (0, 0))


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="max_size"):
        ObservationEncoder(max_size=2)
    with pytest.raises(ValueError, match="n_features"):
        ObservationEncoder(max_size=9, n_features=0)
    with pytest.raises(ValueError, match="value_encoding"):
        ObservationEncoder(max_size=9, value_encoding="nonsense")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value_range"):
        ObservationEncoder(max_size=9, value_range=(1.0, 0.0))


def test_pad_size_decouples_observation_shape_from_maze_size():
    """A size curriculum needs the sampled range to move while the observation
    shape stays fixed; a checkpoint cannot survive a shape change."""
    small = ObservationEncoder(max_size=9, pad_size=25)
    assert small.shape == (25, 25, small.n_channels)

    level = Level(
        walls=OPEN_ROOM,
        agent_start=(1, 1),
        objectives=(Objective((1, 3), value=1.0, feature_id=0), Objective((3, 1), value=0.5, feature_id=1)),
    )
    observation = small.encode(level, (1, 1))
    assert observation.shape == (25, 25, small.n_channels)
    # The 5x5 level still sits top-left, everything beyond it is wall.
    assert observation[5:, :, WALL_CHANNEL].all()


def test_pad_size_must_not_be_smaller_than_max_size():
    with pytest.raises(ValueError, match="smaller than max_size"):
        ObservationEncoder(max_size=25, pad_size=9)


def test_encoder_satisfies_the_observation_protocol():
    from goalmisgen.envs.observation import ObservationScheme

    encoder = ObservationEncoder(max_size=9)
    assert isinstance(encoder, ObservationScheme) or all(hasattr(encoder, name) for name in ("space", "encode", "reset"))
    encoder.reset(make_level())  # no-op, but must exist for stateful encoders
