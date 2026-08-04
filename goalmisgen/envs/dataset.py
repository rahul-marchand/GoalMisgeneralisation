"""Pre-generated level datasets.

Generating a level costs ~3.5 ms — maze construction plus two breadth-first
searches per objective — while stepping one costs ~10 us. Reset is therefore
over 95% of environment time, and pre-generating levels turns it into an array
lookup. This is the same approach the Sokoban work takes with Boxoban, and it
buys reproducible train/validation/test splits as a side effect.

**One dataset serves an entire correlation sweep.** Maze layout, placement,
values and distances are all independent of ``feature_value_correlation``; only
the assignment of ``feature_id`` depends on it, and that is a coin flip applied
at load time by :class:`DatasetLevelSampler`. Features are therefore not
stored.

**Datasets are fingerprinted.** A dataset is only reproducible while the code
that generated it is unchanged, so the fingerprint covers both the sampler
configuration and the source of every module that determines level content.
Loading a dataset whose fingerprint does not match the current code raises,
rather than silently training on a different distribution than intended.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib
import json
import pathlib
from typing import Iterable

import numpy as np

from goalmisgen.envs.features import CorrelatedFeatures, FeatureScheme
from goalmisgen.envs.level import Level, Objective
from goalmisgen.envs.sampling import LevelSampler, MazeLevelSampler
from goalmisgen.envs.solver import objective_distances

CONTENT_MODULES: tuple[str, ...] = ("dataset", "features", "generation", "level", "sampling", "solver", "values")
"""Modules whose source determines what a level contains.

``dataset`` is included because it owns the wall packing, the padding fill and
the block seeding: changing any of those would decode existing files as
*different mazes* while the fingerprint still matched.
"""

# Feature assignment consumes random draws, so a scheme's parameters would
# perturb the stream and hence the layouts. Generation normalises the scheme via
# FeatureScheme.canonical() and discards the resulting features, keeping stored
# content independent of the correlation actually trained on.

ARRAY_FIELDS: tuple[str, ...] = ("walls_packed", "sizes", "agent", "positions", "values", "distances")
"""Arrays persisted to disk, one ``.npy`` each."""

BLOCK_SIZE = 10_000
"""Levels per independently seeded block.

Blocks are seeded by spawning from the master seed, so the dataset depends on
the seed and the block size but *not* on how many worker processes generate it.
Parallelism therefore cannot change the data.
"""


def _without_docstrings(tree: ast.Module) -> ast.Module:
    """The same tree with every docstring removed.

    Docstrings are ordinary string expressions, so they survive into the dumped
    tree and would otherwise invalidate a dataset. They cannot affect a level.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            node.body = node.body[1:]
    return tree


@functools.lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """Hash of the *code* of every module that determines level content.

    The abstract syntax tree is hashed rather than the file bytes, and prose is
    stripped from it, so neither comments nor docstrings invalidate datasets.
    Hashing bytes meant that editing a comment rejected a million-level
    dataset, and a guard that fires on harmless edits is one that ends up
    switched off.

    Computed once per process. The modules that generate levels are already
    imported, so re-reading them from disk describes the working tree rather
    than the code actually running: editing the checkout during a long run made
    the next evaluation reject the dataset the run had been training on for
    hours, which is a report about git, not about the data.
    """
    digest = hashlib.sha256()
    for name in CONTENT_MODULES:
        module = importlib.import_module(f"goalmisgen.envs.{name}")
        assert module.__file__ is not None
        source = pathlib.Path(module.__file__).read_text()
        digest.update(ast.dump(_without_docstrings(ast.parse(source))).encode())
    return digest.hexdigest()[:16]


def canonical_sampler(sampler: MazeLevelSampler) -> MazeLevelSampler:
    """The sampler with its proxy relationship normalised away."""
    return dataclasses.replace(sampler, features=sampler.features.canonical())


def dataset_fingerprint(sampler: MazeLevelSampler) -> str:
    """Identifies the exact distribution a dataset was drawn from."""
    digest = hashlib.sha256()
    digest.update(source_fingerprint().encode())
    digest.update(repr(canonical_sampler(sampler)).encode())
    return digest.hexdigest()[:16]


class FingerprintMismatch(RuntimeError):
    """The dataset on disk was generated by different code or a different config."""


@dataclasses.dataclass(frozen=True)
class LevelDataset:
    """A fixed pool of levels, stored without feature assignments."""

    walls_packed: np.ndarray  # (N, ceil(max_size**2 / 8)) uint8
    sizes: np.ndarray  # (N,) uint8
    agent: np.ndarray  # (N, 2) uint8
    positions: np.ndarray  # (N, K, 2) uint8
    values: np.ndarray  # (N, K) float64 - see below
    distances: np.ndarray  # (N, K) int16, cached solver output
    max_size: int
    fingerprint: str
    path: pathlib.Path | None = None
    """Set when loaded from disk. Enables pickling by reference."""

    stored_splits: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    """Train/validation/test indices written at generation time, if any."""

    def __reduce__(self):
        """Pickle by path when possible.

        Actor processes are forked with the sampler as an argument, so pickling
        the arrays themselves would give every worker its own multi-megabyte
        copy and defeat the memory mapping.
        """
        if self.path is not None:
            return (_load_mapped, (str(self.path), self.fingerprint))
        return super().__reduce__()

    def __len__(self) -> int:
        return len(self.sizes)

    @property
    def n_objectives(self) -> int:
        return self.positions.shape[1]

    def walls(self, index: int) -> np.ndarray:
        """Unpack and crop the stored layout back to its true size."""
        flat = np.unpackbits(self.walls_packed[index], count=self.max_size * self.max_size)
        padded = flat.reshape(self.max_size, self.max_size).astype(np.bool_)
        size = int(self.sizes[index])
        return padded[:size, :size]

    def level(self, index: int, feature_ids: Iterable[int]) -> Level:
        return Level(
            walls=self.walls(index),
            agent_start=(int(self.agent[index, 0]), int(self.agent[index, 1])),
            objectives=tuple(
                Objective(
                    position=(int(self.positions[index, k, 0]), int(self.positions[index, k, 1])),
                    value=float(self.values[index, k]),
                    feature_id=int(feature_id),
                )
                for k, feature_id in enumerate(feature_ids)
            ),
        )

    # ------------------------------------------------------------------
    # Building and persistence
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        sampler: MazeLevelSampler,
        n_levels: int,
        seed: int = 0,
        block_size: int = BLOCK_SIZE,
    ) -> "LevelDataset":
        blocks = [generate_block(*task) for task in block_tasks(sampler, n_levels, seed, block_size)]
        return cls.from_blocks(blocks, sampler)

    @classmethod
    def from_blocks(cls, blocks: list[dict[str, np.ndarray]], sampler: MazeLevelSampler) -> "LevelDataset":
        """Assemble generated blocks. Shared by the serial and parallel paths."""
        return cls(
            walls_packed=np.concatenate([b["walls_packed"] for b in blocks]),
            sizes=np.concatenate([b["sizes"] for b in blocks]),
            agent=np.concatenate([b["agent"] for b in blocks]),
            positions=np.concatenate([b["positions"] for b in blocks]),
            values=np.concatenate([b["values"] for b in blocks]),
            distances=np.concatenate([b["distances"] for b in blocks]),
            max_size=sampler.size_range[1],
            fingerprint=dataset_fingerprint(sampler),
        )

    def save(
        self,
        path: str | pathlib.Path,
        seed: int | None = None,
        block_size: int | None = None,
        splits: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Write as a directory of plain ``.npy`` arrays plus a JSON header.

        Deliberately not a single compressed archive: ``.npy`` files can be
        memory-mapped, so every actor process shares one copy of the levels
        through the page cache. Compressed archives must be decompressed into
        each process's own memory, which at a few hundred workers would cost
        gigabytes for no benefit.
        """
        directory = pathlib.Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        for name in ARRAY_FIELDS:
            np.save(directory / f"{name}.npy", getattr(self, name))

        # Splits belong to the dataset, not to whatever config happens to read
        # it later: recomputing them from config fields means changing a
        # holdout size silently moves levels between train and validation, with
        # no fingerprint to catch it.
        if splits is not None:
            for name, indices in splits.items():
                np.save(directory / f"split_{name}.npy", np.asarray(indices))

        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "max_size": self.max_size,
                    "n_levels": len(self),
                    # Not part of the fingerprint - they do not change the
                    # distribution - but without them a dataset cannot be
                    # regenerated from what is stored beside it.
                    "seed": seed,
                    "block_size": block_size,
                    "splits": sorted(splits) if splits else [],
                },
                indent=2,
            )
        )

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        expected_fingerprint: str | None = None,
        mmap: bool = True,
    ) -> "LevelDataset":
        directory = pathlib.Path(path)
        meta = json.loads((directory / "meta.json").read_text())

        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None) for name in ARRAY_FIELDS}
        dataset = cls(
            **arrays,
            max_size=int(meta["max_size"]),
            fingerprint=str(meta["fingerprint"]),
            path=directory,
        )
        object.__setattr__(
            dataset, "stored_splits", {name: np.load(directory / f"split_{name}.npy") for name in meta.get("splits", [])}
        )

        if expected_fingerprint is not None and dataset.fingerprint != expected_fingerprint:
            raise FingerprintMismatch(
                f"dataset at {path} has fingerprint {dataset.fingerprint!r} but the current "
                f"configuration and code give {expected_fingerprint!r}. The level distribution "
                "has changed; regenerate the dataset or check out the code it was made with."
            )
        return dataset


def _load_mapped(path: str, fingerprint: str) -> "LevelDataset":
    """Reconstruct a memory-mapped dataset in a worker process."""
    return LevelDataset.load(path, expected_fingerprint=fingerprint)


def block_tasks(
    sampler: MazeLevelSampler, n_levels: int, seed: int, block_size: int = BLOCK_SIZE
) -> list[tuple[MazeLevelSampler, np.random.SeedSequence, int]]:
    """Independent generation tasks, one per block.

    Each carries its own spawned seed, so tasks can be run in any order, in any
    number of processes, and produce identical data.
    """
    counts = [block_size] * (n_levels // block_size)
    if n_levels % block_size:
        counts.append(n_levels % block_size)

    children = np.random.SeedSequence(seed).spawn(len(counts))
    normalised = canonical_sampler(sampler)
    return [(normalised, child, count) for child, count in zip(children, counts)]


MAX_STORABLE_SIZE = 255
"""Positions and sizes are stored as uint8, which wraps silently above this."""


def generate_block(sampler: MazeLevelSampler, seed_sequence: np.random.SeedSequence, count: int) -> dict[str, np.ndarray]:
    """Generate one independently seeded block. Safe to run in a worker process."""
    rng = np.random.default_rng(seed_sequence)
    max_size = sampler.size_range[1]
    if max_size > MAX_STORABLE_SIZE:
        raise ValueError(
            f"max_size={max_size} exceeds {MAX_STORABLE_SIZE}: positions are stored as uint8 "
            "and would wrap silently, producing valid-looking levels in the wrong places"
        )
    n_objectives = sampler.n_objectives

    walls_packed = np.empty((count, int(np.ceil(max_size * max_size / 8))), dtype=np.uint8)
    sizes = np.empty(count, dtype=np.uint8)
    agent = np.empty((count, 2), dtype=np.uint8)
    positions = np.empty((count, n_objectives, 2), dtype=np.uint8)
    # float64, not float32: TIE_TOLERANCE is 1e-9 and float32 round-trip error
    # reaches ~3e-8, so storing values in float32 silently changed which
    # objectives counted as tied. With objective_values=(1.0, 0.3) the live
    # sampler found ambiguous levels that the dataset-backed one did not, giving
    # different ground truth for the same configuration.
    values = np.empty((count, n_objectives), dtype=np.float64)
    distances = np.empty((count, n_objectives), dtype=np.int16)

    for index in range(count):
        level = sampler.sample(rng)
        size = level.shape[0]

        padded = np.ones((max_size, max_size), dtype=np.bool_)
        padded[:size, :size] = level.walls
        walls_packed[index] = np.packbits(padded.reshape(-1))

        sizes[index] = size
        agent[index] = level.agent_start
        for k, objective in enumerate(level.objectives):
            positions[index, k] = objective.position
            values[index, k] = objective.value
        distances[index] = [-1 if distance is None else distance for distance in objective_distances(level)]

    return {
        "walls_packed": walls_packed,
        "sizes": sizes,
        "agent": agent,
        "positions": positions,
        "values": values,
        "distances": distances,
    }


@dataclasses.dataclass
class DatasetLevelSampler(LevelSampler):
    """Draws levels from a pre-generated pool, assigning features on the fly.

    Interchangeable with :class:`~goalmisgen.envs.sampling.MazeLevelSampler`, so
    switching an experiment to a fixed level pool is a configuration change.
    """

    dataset: LevelDataset
    features: FeatureScheme = dataclasses.field(default_factory=CorrelatedFeatures)
    indices: np.ndarray | None = None
    """Restrict to a subset, for train/validation/test splits."""

    def __post_init__(self) -> None:
        # The scheme validates its own parameters at construction.
        if self.indices is not None and len(self.indices) == 0:
            raise ValueError("indices is empty; there are no levels to sample")

    def __len__(self) -> int:
        return len(self.dataset) if self.indices is None else len(self.indices)

    def sample(self, rng: np.random.Generator) -> Level:
        position = int(rng.integers(len(self)))
        index = position if self.indices is None else int(self.indices[position])

        values = tuple(float(value) for value in self.dataset.values[index])
        feature_ids = self.features.assign(values, rng)
        return self.dataset.level(index, feature_ids)


def split_indices(n_levels: int, valid: int = 50_000, test: int = 50_000, seed: int = 0) -> dict[str, np.ndarray]:
    """Disjoint train/validation/test index sets, in the style of Boxoban."""
    if valid + test >= n_levels:
        raise ValueError(f"cannot hold out {valid + test} levels from a pool of {n_levels}")

    order = np.random.default_rng(seed).permutation(n_levels)
    return {
        "test": order[:test],
        "valid": order[test : test + valid],
        "train": order[test + valid :],
    }
