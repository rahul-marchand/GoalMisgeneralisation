"""Turning a :class:`~goalmisgen.envs.level.Level` into agent input.

Observations are symbolic channel stacks in ``(height, width, channel)`` order,
which is what cleanba's ``VectorNHWCtoNCHWWrapper`` expects before transposing
to NCHW for the network.

Two properties of this encoding are load-bearing for the experiment.

**Objective channels are keyed by feature, not by index.** If channel order
followed the objective's position in ``Level.objectives`` then, under a fixed
value scheme where objective 0 always holds the higher value, the channel
ordering would itself reveal which objective is valuable. The agent could then
ignore colour entirely and no proxy would ever be learned.

**Values are observable.** Unless the agent can perceive value independently of
colour, colour is the *only* cue to which objective is worth more, and a
colour-following policy is not a misgeneralisation but the sole available
strategy. Value channels are what give the agent a genuine choice of cue, so
that training at high correlation and testing at low correlation actually
distinguishes two hypotheses about what it learned.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import gymnasium as gym
import numpy as np

from goalmisgen.envs.level import Level, Position

ValueEncoding = Literal["none", "at_objective", "broadcast"]

WALL_CHANNEL = 0
AGENT_CHANNEL = 1
FIRST_FEATURE_CHANNEL = 2


@dataclasses.dataclass(frozen=True)
class ObservationEncoder:
    """Encodes a level state as a fixed-shape channel stack.

    Mazes smaller than ``max_size`` are anchored at the top-left and the
    remainder is filled with wall, so every observation has the same shape. This
    is required because cleanba's DRC head flattens before a dense layer, baking
    the input dimensions into the parameter shapes.
    """

    max_size: int
    n_features: int = 2
    value_encoding: ValueEncoding = "at_objective"
    """How objective values are exposed.

    ``at_objective``
        One channel carrying each objective's value at its own cell. Value is
        co-located with the goal, so the network must carry it along with the
        plan — the more interesting case mechanistically, and it keeps value out
        of feature-indexed channels entirely.
    ``broadcast``
        One constant plane per feature, holding that feature's objective value.
        Value is globally available without needing to be transported.
    ``none``
        No value channels. Only appropriate with a single objective; with more,
        colour becomes the only cue to value and the experiment degenerates.
    """

    value_range: tuple[float, float] = (0.0, 1.0)
    """Declared bounds for value channels, used for the observation space."""

    def __post_init__(self) -> None:
        if self.max_size < 3:
            raise ValueError(f"max_size must be at least 3, got {self.max_size}")
        if self.n_features < 1:
            raise ValueError(f"n_features must be at least 1, got {self.n_features}")
        if self.value_encoding not in ("none", "at_objective", "broadcast"):
            raise ValueError(f"unknown value_encoding {self.value_encoding!r}")
        low, high = self.value_range
        if not low < high:
            raise ValueError(f"value_range must be increasing, got {self.value_range}")

    @property
    def n_value_channels(self) -> int:
        if self.value_encoding == "none":
            return 0
        if self.value_encoding == "at_objective":
            return 1
        return self.n_features

    @property
    def n_channels(self) -> int:
        return FIRST_FEATURE_CHANNEL + self.n_features + self.n_value_channels

    @property
    def shape(self) -> tuple[int, int, int]:
        """Observation shape, in ``(height, width, channel)`` order."""
        return (self.max_size, self.max_size, self.n_channels)

    @property
    def first_value_channel(self) -> int:
        return FIRST_FEATURE_CHANNEL + self.n_features

    def space(self) -> gym.spaces.Box:
        low = np.zeros(self.shape, dtype=np.float32)
        high = np.ones(self.shape, dtype=np.float32)
        if self.n_value_channels:
            value_low, value_high = self.value_range
            low[:, :, self.first_value_channel :] = min(0.0, value_low)
            high[:, :, self.first_value_channel :] = value_high
        return gym.spaces.Box(low=low, high=high, shape=self.shape, dtype=np.float32)

    def encode(self, level: Level, agent_position: Position) -> np.ndarray:
        height, width = level.shape
        if height > self.max_size or width > self.max_size:
            raise ValueError(f"level of shape {level.shape} does not fit in max_size={self.max_size}")

        observation = np.zeros(self.shape, dtype=np.float32)

        # Padding counts as wall, so the maze reads as embedded in solid rock.
        observation[:, :, WALL_CHANNEL] = 1.0
        observation[:height, :width, WALL_CHANNEL] = level.walls.astype(np.float32)

        row, col = agent_position
        if not (0 <= row < height and 0 <= col < width) or level.is_wall(agent_position):
            raise ValueError(f"agent_position {agent_position} is not a free cell of the level")
        observation[row, col, AGENT_CHANNEL] = 1.0

        value_low, value_high = self.value_range
        for objective in level.objectives:
            if objective.feature_id >= self.n_features:
                raise ValueError(
                    f"objective has feature_id={objective.feature_id} but the encoder "
                    f"was configured with n_features={self.n_features}"
                )
            if not value_low <= objective.value <= value_high:
                raise ValueError(
                    f"objective value {objective.value} lies outside the " f"declared value_range={self.value_range}"
                )

            objective_row, objective_col = objective.position
            observation[objective_row, objective_col, FIRST_FEATURE_CHANNEL + objective.feature_id] = 1.0

            if self.value_encoding == "at_objective":
                observation[objective_row, objective_col, self.first_value_channel] = objective.value
            elif self.value_encoding == "broadcast":
                observation[:, :, self.first_value_channel + objective.feature_id] = objective.value

        return observation
