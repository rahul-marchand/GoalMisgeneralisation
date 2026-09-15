"""Ore fields: the level distribution the Craftax task is run on.

A Craftax world for the stage-2 task is an ordinary :class:`~goalmisgen.envs.level.Level`
rendered into the engine - walls become stone, free cells grass, objectives ore
tiles - so the level store, the splits, the colour schemes and the value schemes
are the maze's. What differs is the layout generator: an open field with
scattered stone rather than a perfect maze, because that is the shape of the
terrain the engine's own worlds have, and because the maze's single winding
corridor would make the field look like nothing Craftax draws.

**Fingerprints.** Datasets are fingerprinted over the source of
``envs.dataset.CONTENT_MODULES``. This module is deliberately *not* one of
them: adding to that tuple, or to any module in it, would change the fingerprint
of every dataset already on the volume. So ore-field datasets carry their own
fingerprint - the maze one extended by this module's source hash - and are
generated and verified through :func:`generate_ore_fields` and
:func:`ore_field_fingerprint` rather than ``LevelDataset.generate`` and
``dataset_fingerprint``. Existing datasets are untouched by construction.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import pathlib
from collections import deque

import numpy as np

from goalmisgen.envs.dataset import (
    BLOCK_SIZE,
    LevelDataset,
    _without_docstrings,
    block_tasks,
    dataset_fingerprint,
    generate_block,
)
from goalmisgen.envs.generation import validate_shape
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.parallel import worker_pool

_NEIGHBOURS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclasses.dataclass(frozen=True)
class OreFieldGenerator:
    """An open field with a stone border and scattered stone obstacles.

    Every interior cell is stone with probability ``obstacle_density``; the
    largest connected free region is kept and every other free cell is filled,
    so the layout is connected and the sampler's reachability check almost
    never rejects. Shapes must be odd, like the maze's, so the two share one
    size grammar (``validate_shape``).

    Distances in an open field are near-Manhattan, so the distance *gaps* the
    trade-off is measured over are narrower than a maze's; the density is the
    knob that widens them. See the dataset script's printed gap summary.
    """

    obstacle_density: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.obstacle_density < 1.0:
            raise ValueError(f"obstacle_density must be in [0, 1), got {self.obstacle_density}")

    def generate(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        validate_shape(shape)
        walls = np.ones(shape, dtype=np.bool_)
        interior = rng.random((shape[0] - 2, shape[1] - 2)) < self.obstacle_density
        walls[1:-1, 1:-1] = interior
        keep = largest_free_component(walls)
        walls[~keep] = True
        return walls


def largest_free_component(walls: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 4-connected region of free cells."""
    height, width = walls.shape
    seen = np.zeros_like(walls, dtype=np.bool_)
    best: np.ndarray = np.zeros_like(walls, dtype=np.bool_)
    best_size = 0
    for row, col in zip(*np.nonzero(~walls)):
        if seen[row, col]:
            continue
        component = np.zeros_like(walls, dtype=np.bool_)
        queue = deque([(int(row), int(col))])
        seen[row, col] = component[row, col] = True
        size = 0
        while queue:
            r, c = queue.popleft()
            size += 1
            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not walls[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = component[nr, nc] = True
                    queue.append((nr, nc))
        if size > best_size:
            best, best_size = component, size
    return best


@functools.lru_cache(maxsize=1)
def _module_fingerprint() -> str:
    """Hash of this module's code, prose stripped, as ``dataset.source_fingerprint`` does."""
    source = pathlib.Path(__file__).read_text()
    return hashlib.sha256(ast.dump(_without_docstrings(ast.parse(source))).encode()).hexdigest()[:16]


def ore_field_fingerprint(sampler: MazeLevelSampler) -> str:
    """The maze fingerprint extended by the generator's own source.

    ``dataset_fingerprint`` covers the sampler's *configuration* (the generator
    is in its ``repr``) and the source of the maze modules, but not the source
    of a generator defined elsewhere. This adds it.
    """
    if not isinstance(sampler.generator, OreFieldGenerator):
        raise TypeError(f"expected an OreFieldGenerator sampler, got {type(sampler.generator).__name__}")
    digest = hashlib.sha256()
    digest.update(dataset_fingerprint(sampler).encode())
    digest.update(_module_fingerprint().encode())
    return digest.hexdigest()[:16]


def generate_ore_fields(
    sampler: MazeLevelSampler,
    n_levels: int,
    seed: int = 0,
    block_size: int = BLOCK_SIZE,
    workers: int = 1,
) -> LevelDataset:
    """``LevelDataset.generate`` for an ore-field sampler, with the right fingerprint.

    Same blocks, same seeding, same worker independence; only the fingerprint
    the dataset is stamped with differs, so :meth:`LevelDataset.load` with
    ``expected_fingerprint=ore_field_fingerprint(sampler)`` guards it.
    """
    tasks = block_tasks(sampler, n_levels, seed, block_size)
    if workers > 1 and len(tasks) > 1:
        with worker_pool(workers) as pool:
            blocks = pool.starmap(generate_block, tasks)
    else:
        blocks = [generate_block(*task) for task in tasks]
    dataset = LevelDataset.from_blocks(blocks, sampler)
    return dataclasses.replace(dataset, fingerprint=ore_field_fingerprint(sampler))
