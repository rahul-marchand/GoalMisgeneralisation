"""RGB rendering, for figures and debugging only.

Never used as agent input — the agent sees the symbolic channels produced by
:mod:`goalmisgen.envs.observation`. Keeping rendering separate means the colour
scheme can change freely for a paper figure without touching what the network
is trained on.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.envs.level import Level, Position

WALL_COLOUR = (35, 35, 40)
FLOOR_COLOUR = (235, 235, 230)
AGENT_COLOUR = (40, 90, 220)

FEATURE_COLOURS: tuple[tuple[int, int, int], ...] = (
    (220, 60, 50),  # feature 0 — the candidate proxy
    (60, 170, 90),
    (230, 170, 40),
    (150, 80, 200),
)


def render(level: Level, agent_position: Position, cell_pixels: int = 8) -> np.ndarray:
    """Render a level state as an RGB image of shape ``(H*cell, W*cell, 3)``."""
    if cell_pixels < 1:
        raise ValueError(f"cell_pixels must be at least 1, got {cell_pixels}")

    height, width = level.shape
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = FLOOR_COLOUR
    image[level.walls] = WALL_COLOUR

    for objective in level.objectives:
        if objective.feature_id >= len(FEATURE_COLOURS):
            raise ValueError(
                f"no colour defined for feature_id={objective.feature_id}; "
                f"extend FEATURE_COLOURS (has {len(FEATURE_COLOURS)} entries)"
            )
        image[objective.position] = FEATURE_COLOURS[objective.feature_id]

    image[agent_position] = AGENT_COLOUR

    return np.kron(image, np.ones((cell_pixels, cell_pixels, 1), dtype=np.uint8))
