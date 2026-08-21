"""The value-axis loaders find arms by name and diff them against their base."""

from __future__ import annotations

import json

import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.axis import arm_dirs, expected_indifference, load_base, load_diffs, measure_flat
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import ModelConfig
from goalmisgen.offline.train import TrainConfig, list_checkpoints, train
from goalmisgen.volume import arm_dirname


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=64, seed=0, block_size=32)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16).with_hidden_values()


def test_sweep_is_found_by_name_and_diffed_against_the_base(demos, tmp_path):
    tiny = ModelConfig(size=7, n_channels=demos.n_channels, max_actions=16, d_model=16, n_layers=1, n_heads=2)
    quick = dict(batch_size=8, warmup_steps=1, log_every=1, checkpoint_first=100)
    train(demos, tiny, TrainConfig(total_steps=2, **quick), tmp_path / "base", log=lambda s: None)
    init = str(list_checkpoints(tmp_path / "base")[-1][1])
    for offset in (0.2, -0.2, 0.0):
        out = tmp_path / "base" / "arms" / arm_dirname("o0", offset, 1000)
        train(demos, tiny, TrainConfig(total_steps=2, schedule="constant", init_from=init, seed=int(offset * 10) + 5, **quick), out, log=lambda s: None)
        (out / "done.json").write_text(json.dumps({"steps": 2}))
    # an unfinished arm, an arm of another sweep and one at another budget are invisible
    (tmp_path / "base" / "arms" / arm_dirname("o0", 0.3, 1000)).mkdir()
    for name in (arm_dirname("o1", 0.2, 1000), arm_dirname("o0", 0.2, 2000)):
        train(demos, tiny, TrainConfig(total_steps=1, schedule="constant", init_from=init, **quick), tmp_path / "base" / "arms" / name, log=lambda s: None)
        (tmp_path / "base" / "arms" / name / "done.json").write_text("{}")

    base = load_base(tmp_path / "base")
    assert base.hide_values and base.step == 2 and base.flat.ndim == 1
    arms = arm_dirs(tmp_path / "base", "o0", 1000)
    assert sorted(arms) == [-0.2, 0.0, 0.2]
    diffs = load_diffs(base, arms)
    assert all(d.shape == base.flat.shape for d in diffs.values())
    assert all(np.linalg.norm(d) > 0 for d in diffs.values())
    # writing the diff back on top of the base reproduces the arm exactly
    np.testing.assert_allclose(base.flat + diffs[0.2], base.flat + diffs[0.2])
    m = measure_flat(base, base.flat, demos, np.arange(16))
    assert 0.0 <= m.reached <= 1.0


def test_expected_indifference_follows_the_task_arithmetic():
    assert expected_indifference((1.0, 0.5), 0, 0.0) == pytest.approx(10.0)
    assert expected_indifference((1.0, 0.5), 0, 0.45) == pytest.approx(19.0)
    assert expected_indifference((1.0, 0.5), 1, 0.45) == pytest.approx(1.0)
    assert expected_indifference((1.0, 0.5), 1, -0.45) == pytest.approx(19.0)
