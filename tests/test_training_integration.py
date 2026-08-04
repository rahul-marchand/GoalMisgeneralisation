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

from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33, maze_smoke_test
from goalmisgen.configs.writers import CsvWriter

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


def test_smoke_preset_is_a_tiny_task_not_a_scientific_one():
    """Its budget is sized for the GPU learning assertion; the CPU tests
    override total_timesteps to a few hundred steps of their own."""
    args = maze_smoke_test()
    assert args.train_env.max_size == 5
    assert args.eval_envs == {}
    assert args.total_timesteps < maze_drc33().total_timesteps / 100


# --------------------------------------------------------------------------
# The gate: does it actually learn?
# --------------------------------------------------------------------------


RETURN_METRIC = "charts/0/avg_episode_returns"


def run_training(args) -> CsvWriter:
    import cleanba.cleanba_impala
    from cleanba.cleanba_impala import train

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CsvWriter(args, Path(tmpdir))
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


@pytest.mark.slow
def test_training_survives_an_evaluation_pass(tmp_path):
    """Run training long enough to trigger an evaluation.

    Evaluation exercises code paths training never touches, and a failure there
    surfaces on the evaluation thread — so the process hangs instead of
    crashing, which reads as a stall while still costing GPU time. This is the
    only test that would catch a Sokoban assumption in cleanba's eval.
    """
    from cleanba.evaluate import EvalConfig

    from goalmisgen.envs.dataset import LevelDataset

    base = MazeConfig(max_episode_steps=30, min_size=5, max_size=5)
    LevelDataset.generate(base.live_sampler(), n_levels=200, seed=0, block_size=100).save(tmp_path / "levels")

    args = maze_smoke_test()
    args.net = dataclasses.replace(args.net, n_recurrent=1, repeats_per_step=1)
    args.local_num_envs = 8
    args.num_steps = 8
    args.total_timesteps = 8 * 8 * 8

    train_env = dataclasses.replace(
        args.train_env,
        num_envs=8,
        asynchronous=False,
        level_dataset=str(tmp_path / "levels"),
        dataset_valid_levels=50,
        dataset_test_levels=50,
    )
    args.train_env = train_env
    args.eval_envs = {
        "rho000": EvalConfig(
            dataclasses.replace(train_env, num_envs=4, feature_value_correlation=0.0, dataset_split="valid"),
            n_episode_multiple=1,
            steps_to_think=[0],
        )
    }
    args.eval_at_steps = frozenset([1])
    args.save_model = False

    writer = run_training(args)

    evaluated = [c for c in writer.metrics.columns if "rho000" in c]
    assert evaluated, f"evaluation never ran; columns were {list(writer.metrics.columns)[:10]}"


@pytest.mark.slow
def test_checkpointing_works_with_the_csv_writer(tmp_path):
    """Save a real checkpoint.

    The writer interface is only partly exercised by training: `add_scalar` runs
    constantly, but `step_digits` and `_save_dir` are touched only when a
    checkpoint is written. A writer missing them trains for hundreds of
    thousands of steps and then dies — which has cost us a run.
    """
    args = maze_smoke_test()
    args.net = dataclasses.replace(args.net, n_recurrent=1, repeats_per_step=1)
    args.local_num_envs = 8
    args.num_steps = 8
    args.train_env = dataclasses.replace(args.train_env, num_envs=8, asynchronous=False)
    args.total_timesteps = 8 * 8 * 8
    args.save_model = True
    args.base_run_dir = tmp_path
    # Checkpoints are written when the learner version reaches an eval step, so
    # a run with eval_at_steps empty never exercises the save path at all.
    args.eval_at_steps = frozenset([2])

    import cleanba.cleanba_impala
    from cleanba.cleanba_impala import train

    writer = CsvWriter(args, tmp_path)
    cleanba.cleanba_impala.MUST_STOP_PROGRAM = False
    train(args, writer=writer)
    writer.flush()

    checkpoints = list((tmp_path / "local-files").glob("cp_*"))
    assert checkpoints, f"no checkpoint written; dir held {list((tmp_path / 'local-files').iterdir())}"
    assert (tmp_path / "metrics.csv").exists()
    assert not writer.metrics.empty


def test_csv_writer_exposes_the_full_writer_interface():
    """Cheap guard against the next missing attribute."""
    args = maze_smoke_test()
    writer = CsvWriter(args, Path("/tmp/goalmisgen-writer-check"))
    for attribute in ("step_digits", "named_save_dir", "_save_dir", "add_scalar", "save_dir"):
        assert hasattr(writer, attribute), f"CsvWriter is missing {attribute}"
    assert writer.step_digits >= 1


def test_discount_horizon_exceeds_the_episode_limit():
    """A goal the agent cannot see is a goal it cannot learn.

    With terminal-only reward, a discount horizon shorter than the episode makes
    distant objectives nearly invisible to the value function while the step
    penalty stays immediate. The inherited gamma of 0.97 gives a 33-step horizon
    against a 120-step limit, which discounts a goal at the 90th-percentile
    distance for 25x25 mazes to 0.08.
    """
    args = maze_drc33()
    horizon = 1.0 / (1.0 - args.loss.gamma)
    assert horizon > args.train_env.max_episode_steps, (
        f"discount horizon {horizon:.0f} is shorter than the {args.train_env.max_episode_steps}-step "
        "episode limit; distant objectives will not be learnable"
    )


def test_gradient_clipping_is_not_the_two_billion_step_value():
    """Guard against silently re-inheriting sokoban_resnet59's clip."""
    assert maze_drc33().max_grad_norm > 1e-3


def test_evaluation_control_arm_tracks_chance_not_a_fixed_half():
    """With three objectives, 0.5 is a correlated condition, not a control."""
    from goalmisgen.configs.presets import default_eval_correlations

    assert default_eval_correlations(2) == (1.0, 0.5, 0.0)
    assert default_eval_correlations(3) == pytest.approx((1.0, 1 / 3, 0.0))

    three = maze_drc33(n_objectives=3, objective_values=(1.0, 0.6, 0.3))
    arms = sorted(e.env.feature_value_correlation for e in three.eval_envs.values())
    assert arms == pytest.approx([0.0, 1 / 3, 1.0])


def test_build_env_gives_analysis_the_same_environment_training_saw():
    config = MazeConfig(max_episode_steps=40, min_size=5, max_size=9, n_objectives=2)
    env = config.build_env()
    assert env.step_limit == 40
    assert env.step_penalty == config.step_penalty
    assert env.observation_space.shape == (9, 9, config.encoder().n_channels)


def test_a_feature_palette_wider_than_the_objective_count_is_rejected():
    """It was accepted, and did nothing.

    ``CorrelatedFeatures`` assigns a permutation of one feature per objective,
    so the extra channels were always zero: the experiment this was meant to
    enable — 'did the agent learn *this* colour, or whichever colour is first' —
    silently was not running. Supporting it needs a scheme that draws from the
    wider palette, so until then the configuration must fail rather than lie.
    """
    with pytest.raises(ValueError, match="dead channels"):
        MazeConfig(max_episode_steps=40, n_objectives=2, n_features=4)

    assert MazeConfig(max_episode_steps=40, n_objectives=2).encoder().n_features == 2


def test_the_csv_writer_survives_concurrent_logging(tmp_path):
    """The rollout and learner threads both log, and enlarging a DataFrame
    through .loc is not atomic.

    A barrier is needed to provoke it: the race is rare per update but a run
    makes tens of thousands, and the failure lands on the rollout thread, where
    an exception stalls the learner until its queue times out.
    """
    import threading

    writer = CsvWriter(maze_smoke_test(), tmp_path, flush_every=5)

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def log(prefix: str, count: int) -> None:
        barrier.wait()
        try:
            for step in range(count):
                writer.add_scalar(f"{prefix}/metric_{step % 7}", float(step), step * 640)
        except BaseException as error:  # noqa: BLE001 - the point is to observe any failure
            errors.append(error)

    threads = [threading.Thread(target=log, args=(name, 60)) for name in ("rollout", "learner")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent logging raised: {errors[0]!r}"

    writer.flush()
    reloaded = pd.read_csv(writer.csv_path, index_col=0)
    assert len(reloaded) > 0, "the frame must survive in a readable state"
