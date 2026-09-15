"""End to end: the route model trains on Craftax demonstrations and is scored by the engine.

Marked slow, like the maze's smoke test it mirrors. The bar is the same
modest one - the pieces fit - but every route the model emits is executed by
``craftax_step``, so a 17-way head that learned the wrong action ids, or a
DO in the wrong place, shows up here and nowhere earlier.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from goalmisgen.craftax.demos import CraftaxDemoSet
from goalmisgen.craftax.levels import OreFieldGenerator, generate_ore_fields
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.axis import arm_dirs
from goalmisgen.offline.decode import evaluate, greedy_decode
from goalmisgen.offline.demonstrations import load_demonstrations
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.train import TrainConfig, list_checkpoints, load_run_config, train
from goalmisgen.volume import arm_dirname

SIZE = 9
TINY = ModelConfig(size=SIZE, n_channels=4, n_actions=17, max_actions=24, d_model=32, n_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def demos(tmp_path_factory) -> CraftaxDemoSet:
    sampler = MazeLevelSampler(generator=OreFieldGenerator(0.25), size_range=(SIZE, SIZE))
    dataset = generate_ore_fields(sampler, n_levels=3000, seed=0, block_size=1500, workers=2)
    demos = CraftaxDemoSet.generate(dataset, np.arange(3000), rho=1.0, step_limit=60, max_actions=24, workers=2)
    path = tmp_path_factory.mktemp("demos") / "train"
    demos.with_hidden_values().save(path)
    return load_demonstrations(path, hide_values=True)


@pytest.mark.slow
def test_smoke_training_learns_routes_the_engine_accepts(demos, tmp_path):
    assert isinstance(demos, CraftaxDemoSet) and demos.n_channels == 4
    held_out = np.arange(2800, 3000)
    seen = np.arange(200)
    train_set = demos.subset(np.arange(2800))
    config = TrainConfig(
        total_steps=1000,
        batch_size=64,
        learning_rate=3e-3,
        warmup_steps=50,
        log_every=100,
        checkpoint_first=50,
        checkpoint_ratio=4.0,
    )
    rows = []

    def evaluator(params, step):
        summary, _, _ = evaluate(RoutePrefixLM(TINY), params, demos, held_out)
        rows.append((step, summary))
        return summary.as_row()

    params = train(train_set, TINY, config, tmp_path / "base", evaluate=evaluator, log=lambda s: None)
    config_json = load_run_config(tmp_path / "base")
    assert config_json["model"]["n_actions"] == 17 and config_json["demos"]["task"]["task"] == "craftax-ore"

    metrics = (tmp_path / "base" / "metrics.csv").read_text().splitlines()
    first, last = float(metrics[1].split(",")[1]), float(metrics[-1].split(",")[1])
    assert last < 0.7 * first, f"loss did not fall: {first:.3f} -> {last:.3f}"

    model = RoutePrefixLM(TINY)
    on_seen, decoded, outcomes = evaluate(model, params, demos, seen)
    print(f"\nuntrained (held out): {rows[0][1]}\ntrained (held out):   {rows[-1][1]}\ntrained (seen):       {on_seen}")
    assert on_seen.behaviour.reached_objective > 0.5
    assert on_seen.legal > 0.5 and on_seen.emitted_eos > 0.9
    assert rows[-1][1].behaviour.reached_objective > rows[0][1].behaviour.reached_objective
    # Every emitted route was executed by the engine, and the engine's extras are present.
    assert all("walls_mined" in o and "positions" in o for o in outcomes)
    ends_in_do = [decoded.actions[i, decoded.lengths[i] - 1] == 5 for i in range(len(seen)) if decoded.lengths[i] > 0]
    assert np.mean(ends_in_do) > 0.5, "learned routes end with DO"

    # The arm grammar the value-axis analysis discovers: an arm directory with done.json and a checkpoint.
    arm = tmp_path / "base" / "arms" / arm_dirname("o1", 0.2, 1000)
    train(
        train_set,
        TINY,
        TrainConfig(total_steps=10, batch_size=16, warmup_steps=1, checkpoint_first=100),
        arm,
        log=lambda s: None,
    )
    (arm / "done.json").write_text(json.dumps({"steps": 1000}))
    assert list(arm_dirs(tmp_path / "base", "o1", 1000)) == [0.2]

    _, loaded = __import__("goalmisgen.offline.train", fromlist=["load_checkpoint"]).load_checkpoint(
        list_checkpoints(tmp_path / "base")[-1][1]
    )
    a = greedy_decode(model, loaded, demos.observations(held_out[:8]))
    b = greedy_decode(model, params, demos.observations(held_out[:8]))
    np.testing.assert_array_equal(a.actions, b.actions)
