"""The policy networks that stand in for the DRC, and the readers that probe them.

The question these guard is whether a result is a property of the task or of
the DRC it was found on, so the tests are about the *contract*: each network
trains in the same loop, saves and loads through the same checkpoint path,
and hands the same kind of per-cell state to the same probes.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import farconf
import jax
import numpy as np
import pytest
from cleanba.config import Args

from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33, maze_resnet, maze_transformer, preset_for
from goalmisgen.nets.transformer import TransformerSpec

SIZE = 7
PRESETS = {"drc33": maze_drc33, "resnet": maze_resnet, "vit": maze_transformer}


def small_env(num_envs: int = 2, seed: int = 0) -> MazeConfig:
    return MazeConfig(max_episode_steps=40, num_envs=num_envs, min_size=SIZE, max_size=SIZE, asynchronous=False, seed=seed)


def small_preset(name: str):
    args = PRESETS[name](min_size=SIZE, max_size=SIZE)
    if name == "drc33":
        # One layer, one tick: enough to exercise the recurrent path on CPU.
        args.net = dataclasses.replace(args.net, n_recurrent=1, repeats_per_step=1)
    return args


@pytest.fixture(scope="module")
def policies():
    envs = small_env().make()
    out = {}
    for name in PRESETS:
        args = small_preset(name)
        out[name] = args.net.init_params(envs, jax.random.PRNGKey(0))
    return out


# --------------------------------------------------------------------------
# Presets: only the network changes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["resnet", "vit"])
def test_swapped_presets_differ_from_the_drc_in_the_network_alone(name):
    drc = maze_drc33(feature_value_correlation=0.5, level_dataset="/some/levels")
    swapped = PRESETS[name](feature_value_correlation=0.5, level_dataset="/some/levels")

    assert type(swapped.net).__name__ != type(drc.net).__name__
    for field in dataclasses.fields(Args):
        if field.name in ("net",):
            continue
        assert getattr(swapped, field.name) == getattr(drc, field.name), f"{field.name} differs"


def test_preset_lookup_by_short_name():
    assert preset_for("resnet") is maze_resnet
    assert preset_for("vit") is maze_transformer
    assert preset_for("drc33") is maze_drc33
    with pytest.raises(ValueError, match="unknown network"):
        preset_for("lstm")


def test_the_resnet_is_cleanbas_own_and_unmodified():
    from cleanba.network import GuezResNetConfig

    net = maze_resnet().net
    assert type(net) is GuezResNetConfig
    assert net.normalize_input is False and net.yang_init is False
    assert len(set(net.channels)) == 1, "equal widths so layer_slice divides the probe state evenly"


# --------------------------------------------------------------------------
# The transformer itself
# --------------------------------------------------------------------------


def test_transformer_is_a_non_recurrent_policy(policies):
    policy, carry, params = policies["vit"]
    assert carry == (), "a non-recurrent network must hand cleanba an empty carry"
    envs = small_env().make()
    obs, _ = envs.reset(seed=0)
    _, action, logits, _ = policy.apply(
        params, carry, obs, np.zeros(2, dtype=bool), jax.random.PRNGKey(0), method=policy.get_action, temperature=0.0
    )
    assert logits.shape == (2, 4) and action.shape == (2,)


def test_transformer_records_one_residual_grid_per_layer_plus_the_embedding(policies):
    policy, carry, params = policies["vit"]
    envs = small_env().make()
    obs, _ = envs.reset(seed=0)
    _, variables = policy.apply(
        params,
        carry,
        obs,
        np.zeros(2, dtype=bool),
        jax.random.PRNGKey(0),
        method=policy.get_action,
        temperature=0.0,
        mutable=["intermediates"],
    )
    grids = variables["intermediates"]["network_params"]["residual"]
    spec: TransformerSpec = policy.cfg
    assert len(grids) == spec.n_layers + 1
    assert all(g.shape == (2, SIZE, SIZE, spec.d_model) for g in grids)


def test_transformer_spec_survives_the_checkpoint_config_round_trip():
    """``load_train_state`` rebuilds the network from cfg.json via farconf.

    A spec it cannot serialise trains for hours and then cannot be loaded.
    """
    args = maze_transformer(min_size=SIZE, max_size=SIZE)
    rebuilt = farconf.from_dict(farconf.to_dict(args, Args), Args)
    assert isinstance(rebuilt.net, TransformerSpec)
    assert rebuilt.net == args.net


# --------------------------------------------------------------------------
# Training: the same loop, the same checkpoint path, the same evaluation
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["resnet", "vit"])
def test_swapped_network_trains_evaluates_and_reloads(tmp_path, name):
    """A few updates on CPU, through an evaluation pass and a checkpoint.

    Three things that training alone would not catch: the evaluation thread
    (a failure there hangs the process rather than crashing it), the save path,
    and ``load_train_state`` rebuilding our own ``PolicySpec`` from cfg.json.
    """
    import cleanba.cleanba_impala
    from cleanba.cleanba_impala import load_train_state, train
    from cleanba.evaluate import EvalConfig

    from goalmisgen.configs.presets import maze_smoke_test
    from goalmisgen.configs.writers import CsvWriter
    from goalmisgen.envs.dataset import LevelDataset

    base = MazeConfig(max_episode_steps=30, min_size=5, max_size=5)
    LevelDataset.generate(base.live_sampler(), n_levels=200, seed=0, block_size=100).save(tmp_path / "levels")

    args = maze_smoke_test()
    args.net = PRESETS[name](min_size=5, max_size=5).net
    if name == "vit":
        args.net = dataclasses.replace(args.net, n_layers=2, d_model=32, n_heads=2)
    else:
        args.net = dataclasses.replace(args.net, channels=(16,) * 3, kernel_sizes=(3,) * 3, strides=(1,) * 3)
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
    args.eval_at_steps = frozenset([2])
    args.save_model = True
    args.base_run_dir = tmp_path / "run"

    writer = CsvWriter(args, args.base_run_dir)
    cleanba.cleanba_impala.MUST_STOP_PROGRAM = False
    train(args, writer=writer)
    writer.flush()

    returns = writer.metrics["charts/0/avg_episode_returns"].dropna()
    assert len(returns) >= 3 and np.isfinite(returns).all()
    assert [c for c in writer.metrics.columns if "rho000" in c], "evaluation never ran"

    checkpoints = sorted((args.base_run_dir / "local-files").glob("cp_*"))
    assert checkpoints, "no checkpoint written"
    policy, carry, loaded_args, train_state, _ = load_train_state(checkpoints[-1], env_cfg=train_env)
    assert type(loaded_args.net) is type(args.net)
    assert carry == ()
