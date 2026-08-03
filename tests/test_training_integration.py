"""End-to-end training tests.

These are the gate before any GPU spend: they prove the maze environment, the
cleanba IMPALA loop and JAX fit together, and that a DRC agent's return actually
improves on the task. Everything upstream is unit-tested in isolation; only this
file exercises the whole stack.

Marked ``slow`` because training even a tiny agent takes tens of seconds. Run
the fast suite with ``-m 'not slow'``.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cleanba.cleanba_impala import WandbWriter

from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33, maze_smoke_test


class DataFrameWriter(WandbWriter):
    """Collects metrics in memory instead of sending them to W&B.

    Mirrors the helper in the upstream test suite (``tests/test_cartpole.py``).
    """

    def __init__(self, cfg, save_dir: Path):
        self.metrics = pd.DataFrame()
        self.states: dict = {}
        self._save_dir = save_dir
        # WandbWriter sets this in __init__; the upstream helper omits it because
        # it predates the checkpoint-resume path, which lists this directory.
        self.named_save_dir = save_dir
        (save_dir / "local-files").mkdir(parents=True, exist_ok=True)

    def add_scalar(self, name: str, value, global_step: int) -> None:
        try:
            values = list(value)
        except TypeError:
            self.metrics.loc[global_step, name] = value
            return

        for offset, item in enumerate(values):
            try:
                self.metrics.loc[global_step + 640 * offset, name] = item.item()
            except (TypeError, AttributeError, ValueError):
                self.states[global_step + 640 * offset, name] = value


# --------------------------------------------------------------------------
# Configuration wiring (fast)
# --------------------------------------------------------------------------


def test_vectorised_env_produces_nchw_observations():
    config = MazeConfig(max_episode_steps=60, num_envs=4, min_size=5, max_size=9, asynchronous=False)
    envs = config.make()

    expected_channels = config.encoder().n_channels
    assert envs.single_observation_space.shape == (expected_channels, 9, 9)

    observation, _ = envs.reset(seed=0)
    assert observation.shape == (4, expected_channels, 9, 9)
    assert observation.dtype == np.float32


def test_ground_truth_survives_vector_batching():
    """Info values must be batchable, or evaluation loses the ground truth."""
    config = MazeConfig(max_episode_steps=60, num_envs=4, min_size=5, max_size=9, asynchronous=False)
    envs = config.make()
    _, info = envs.reset(seed=0)
    for key in ("optimal_index", "optimal_feature_id", "utility_margin"):
        assert key in info


def test_config_rejects_an_unmeasurable_setup():
    with pytest.raises(ValueError, match="colour the only cue"):
        MazeConfig(max_episode_steps=60, n_objectives=2, value_encoding="none")


def test_config_rejects_mismatched_value_counts():
    with pytest.raises(ValueError, match="objective_values"):
        MazeConfig(max_episode_steps=60, n_objectives=3, objective_values=(1.0, 0.5))


def test_preset_builds_a_drc33_network_and_shifted_eval_envs():
    args = maze_drc33(feature_value_correlation=1.0)

    assert type(args.net).__name__ == "ConvLSTMConfig"
    assert args.net.n_recurrent == 3 and args.net.repeats_per_step == 3

    assert isinstance(args.train_env, MazeConfig)
    assert args.train_env.feature_value_correlation == 1.0

    # Evaluation differs from training only in the correlation.
    correlations = {cfg.env.feature_value_correlation for cfg in args.eval_envs.values()}
    assert correlations == {1.0, 0.5, 0.0}
    assert set(args.eval_envs) == {"rho100", "rho050", "rho000"}


def test_smoke_preset_is_small_enough_to_run_on_cpu():
    args = maze_smoke_test()
    assert args.total_timesteps <= 100_000
    assert args.eval_envs == {}
    assert args.train_env.max_size == 5


# --------------------------------------------------------------------------
# The gate: does it actually learn?
# --------------------------------------------------------------------------


RETURN_METRIC = "charts/0/avg_episode_returns"


def run_training(args) -> DataFrameWriter:
    import cleanba.cleanba_impala
    from cleanba.cleanba_impala import train

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DataFrameWriter(args, save_dir=Path(tmpdir))
        cleanba.cleanba_impala.MUST_STOP_PROGRAM = False
        train(args, writer=writer)
    return writer


@pytest.mark.slow
def test_training_runs_end_to_end():
    """A handful of updates on a DRC(1,1), purely to prove the stack fits together.

    This is the cheap gate: it catches shape, dtype and interface breakage
    between the environment, cleanba and JAX for a few seconds of CPU. It
    deliberately does *not* assert learning — at ~100 steps/second on CPU, a
    run long enough to show improvement takes tens of minutes, which belongs on
    a GPU rather than in the test suite.
    """
    args = maze_smoke_test()
    args.net = dataclasses.replace(args.net, n_recurrent=1, repeats_per_step=1)
    args.local_num_envs = 8
    args.num_steps = 8
    args.train_env = dataclasses.replace(args.train_env, num_envs=8)
    args.total_timesteps = 8 * 8 * 6

    writer = run_training(args)

    returns = writer.metrics[RETURN_METRIC].dropna()
    assert len(returns) >= 3, f"training produced too few updates: {len(returns)}"
    assert np.isfinite(returns).all(), f"non-finite returns: {returns.tolist()}"

    lengths = writer.metrics["charts/0/returned_avg_episode_length"].dropna()
    assert (lengths > 0).all(), "episodes never completed"


@pytest.mark.gpu
def test_a_drc_agent_learns_on_the_maze():
    """Require the return to actually improve. Needs a GPU to finish in reasonable time.

    Run with: uv run pytest -m gpu
    """
    writer = run_training(maze_smoke_test())

    returns = writer.metrics[RETURN_METRIC].dropna()
    assert len(returns) >= 6, f"expected several logged returns, got {len(returns)}"

    third = max(1, len(returns) // 3)
    first, last = returns.iloc[:third].mean(), returns.iloc[-third:].mean()
    assert last > first, f"return did not improve: {first:.3f} -> {last:.3f}"


# --------------------------------------------------------------------------
# Dataset-backed environments
# --------------------------------------------------------------------------


def test_config_can_be_backed_by_a_level_dataset(tmp_path):
    from goalmisgen.envs.dataset import LevelDataset

    base = MazeConfig(max_episode_steps=60, min_size=5, max_size=9)
    LevelDataset.generate(base.live_sampler(), n_levels=300, seed=0, block_size=100).save(tmp_path / "levels")

    config = dataclasses.replace(
        base,
        num_envs=4,
        asynchronous=False,
        level_dataset=str(tmp_path / "levels"),
        dataset_valid_levels=50,
        dataset_test_levels=50,
    )
    envs = config.make()
    observation, info = envs.reset(seed=0)

    assert observation.shape == (4, config.encoder().n_channels, 9, 9)
    assert "optimal_index" in info


def test_a_stale_dataset_is_refused(tmp_path):
    from goalmisgen.envs.dataset import FingerprintMismatch, LevelDataset

    base = MazeConfig(max_episode_steps=60, min_size=5, max_size=9)
    LevelDataset.generate(base.live_sampler(), n_levels=200, seed=0, block_size=100).save(tmp_path / "levels")

    # A different level distribution must not silently reuse these levels.
    mismatched = dataclasses.replace(
        base,
        step_penalty=base.step_penalty,
        objective_values=(1.0, 0.25),
        level_dataset=str(tmp_path / "levels"),
        dataset_valid_levels=25,
        dataset_test_levels=25,
    )
    with pytest.raises(FingerprintMismatch):
        mismatched.sampler()


def test_splits_are_honoured(tmp_path):
    from goalmisgen.envs.dataset import LevelDataset

    base = MazeConfig(max_episode_steps=60, min_size=5, max_size=9)
    LevelDataset.generate(base.live_sampler(), n_levels=300, seed=0, block_size=100).save(tmp_path / "levels")

    sizes = {}
    for split in ("train", "valid", "test"):
        config = dataclasses.replace(
            base,
            level_dataset=str(tmp_path / "levels"),
            dataset_split=split,
            dataset_valid_levels=50,
            dataset_test_levels=50,
        )
        sizes[split] = len(config.sampler())

    assert sizes == {"train": 200, "valid": 50, "test": 50}


def test_unknown_split_is_rejected(tmp_path):
    from goalmisgen.envs.dataset import LevelDataset

    base = MazeConfig(max_episode_steps=60, min_size=5, max_size=9)
    LevelDataset.generate(base.live_sampler(), n_levels=200, seed=0, block_size=100).save(tmp_path / "levels")
    config = dataclasses.replace(
        base,
        level_dataset=str(tmp_path / "levels"),
        dataset_split="nonsense",
        dataset_valid_levels=25,
        dataset_test_levels=25,
    )
    with pytest.raises(ValueError, match="unknown dataset_split"):
        config.sampler()


def test_evaluation_never_draws_from_the_training_split(tmp_path):
    """Evaluating on training levels would confound misgeneralisation with memorisation."""
    from goalmisgen.envs.dataset import LevelDataset

    base = MazeConfig(max_episode_steps=60, min_size=5, max_size=9)
    LevelDataset.generate(base.live_sampler(), n_levels=300, seed=0, block_size=100).save(tmp_path / "levels")

    args = maze_drc33(min_size=5, max_size=9, level_dataset=str(tmp_path / "levels"))
    assert args.train_env.dataset_split == "train"
    for evaluation in args.eval_envs.values():
        assert evaluation.env.dataset_split == "valid"


def test_preset_passes_the_dataset_to_every_environment(tmp_path):
    args = maze_drc33(level_dataset="/some/path")
    assert args.train_env.level_dataset == "/some/path"
    assert all(e.env.level_dataset == "/some/path" for e in args.eval_envs.values())


def test_preset_without_a_dataset_still_samples_live():
    args = maze_drc33()
    assert args.train_env.level_dataset is None
