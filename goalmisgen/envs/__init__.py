"""Maze environments for studying goal misgeneralisation."""

from gymnasium.envs.registration import register

from goalmisgen.envs.generation import MazeGenerator, RecursiveBacktracker
from goalmisgen.envs.level import Level, Objective, Position
from goalmisgen.envs.maze import ACTION_NAMES, MazeEnv
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import LevelSampler, MazeLevelSampler
from goalmisgen.envs.solver import LevelSolution, distance_field, shortest_path, solve
from goalmisgen.envs.values import FixedValues, UniformValues, ValueScheme

__all__ = [
    "ACTION_NAMES",
    "FixedValues",
    "Level",
    "LevelSampler",
    "LevelSolution",
    "MazeEnv",
    "MazeGenerator",
    "MazeLevelSampler",
    "Objective",
    "ObservationEncoder",
    "Position",
    "RecursiveBacktracker",
    "UniformValues",
    "ValueScheme",
    "distance_field",
    "shortest_path",
    "solve",
]

# Truncation is handled inside MazeEnv rather than by a TimeLimit wrapper, so
# max_episode_steps is deliberately not registered here.
register(id="Maze-v0", entry_point="goalmisgen.envs.maze:MazeEnv")
