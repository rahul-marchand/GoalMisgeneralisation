"""Training configuration for maze experiments."""

from goalmisgen.configs import compat
from goalmisgen.configs.env import MazeConfig

# Applied on import: parts of cleanba's evaluation assume Sokoban observations
# and would otherwise kill the evaluation thread, hanging training.
compat.install()

__all__ = ["MazeConfig", "compat"]
