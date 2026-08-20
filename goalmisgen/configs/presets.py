"""Training presets for maze experiments.

``maze_drc33`` inherits far.ai's DRC(3,3) network and optimiser settings from
``boxworld_drc33`` and swaps in the maze.

Three loss/optimiser values are deliberately *not* inherited, because
``boxworld_drc33`` derives them from ``sokoban_resnet59`` — a ResNet
configuration tuned for a two-billion-step Sokoban run:

===================  ==========  =========  ===================================
value                inherited   ours       why
===================  ==========  =========  ===================================
``gamma``            0.97        0.995      Sokoban's reward is dense; ours is
                                            terminal-only. The discount horizon
                                            must exceed the episode limit.
``vtrace_lambda``    0.5         0.97       far.ai's own DRC value.
``max_grad_norm``    2.5e-4      0.015      Theirs suits 2e9 steps; 0.015 is
                                            their DRC-on-Sokoban value.
===================  ==========  =========  ===================================

Everything else — architecture, learning rate, batch shape — is inherited, so
results remain attributable to the task rather than to tuning we invented.
"""

from __future__ import annotations

import dataclasses

from cleanba.config import Args, boxworld_drc33
from cleanba.evaluate import EvalConfig
from cleanba.network import GuezResNetConfig, IdentityNorm

from goalmisgen.configs.env import MazeConfig
from goalmisgen.nets.transformer import TransformerSpec


def default_eval_correlations(n_objectives: int) -> tuple[float, ...]:
    """Training correlation, chance, and fully reversed.

    The middle arm is the *control*, and chance is ``1 / n_objectives`` rather
    than 0.5 - with three objectives a fixed 0.5 would be a positively
    correlated condition dressed up as a control. Tracking all three through
    training shows when a proxy is adopted, not just whether it survived.
    """
    return (1.0, 1.0 / n_objectives, 0.0)


EVAL_SEED_OFFSET = 1_000_000
"""Separation between the training and evaluation seeds.

A vector environment seeds sub-environment *i* with ``seed + i``, so an
evaluation seed of ``seed + 1`` put every evaluation level inside the training
seed range whenever levels are sampled live - evaluating on training levels,
which confounds misgeneralisation with memorisation. The offset must exceed any
plausible environment count.
"""


def maze_drc33(
    feature_value_correlation: float = 1.0,
    eval_correlations: tuple[float, ...] | None = None,
    min_size: int = 5,
    max_size: int = 25,
    n_objectives: int = 2,
    objective_values: tuple[float, ...] | None = None,
    step_penalty: float = 0.05,
    randomise_values: bool = False,
    hide_values: bool = False,
    max_episode_steps: int = 120,
    total_timesteps: int = 200_000_000,
    seed: int = 1234,
    level_dataset: str | None = None,
    gamma: float = 0.995,
    max_grad_norm: float = 0.015,
    eval_num_envs: int = 64,
    eval_steps_to_think: tuple[int, ...] = (0,),
) -> Args:
    """DRC(3,3) on multi-objective mazes.

    Evaluation environments differ from training in ``feature_value_correlation``
    — the misgeneralisation measurement — and, when a level dataset is used, in
    which split they draw from. Evaluating on training levels would confound
    misgeneralisation with memorisation, so the two must never share levels.
    """
    eval_correlations = eval_correlations or default_eval_correlations(n_objectives)

    out = boxworld_drc33()

    def env(correlation: float, split: str, **overrides) -> MazeConfig:
        settings = dict(
            max_episode_steps=max_episode_steps,
            num_envs=1,
            min_size=min_size,
            max_size=max_size,
            n_objectives=n_objectives,
            step_penalty=step_penalty,
            randomise_values=randomise_values,
            value_encoding="none" if hide_values else "at_objective",
            colour_is_the_only_value_cue=hide_values,
            feature_value_correlation=correlation,
            **({"objective_values": objective_values} if objective_values else {}),
            level_dataset=level_dataset,
            dataset_split=split,
            seed=seed,
        )
        settings.update(overrides)
        return MazeConfig(**settings)  # type: ignore[arg-type]

    out.train_env = env(feature_value_correlation, split="train")
    out.eval_envs = {
        f"rho{int(round(correlation * 100)):03d}": EvalConfig(
            # Synchronous, and far fewer environments than upstream uses.
            # Evaluation runs on the rollout thread, so while it works the
            # learner cannot hand over parameters and its queue backs up; with a
            # fast environment the learner laps the evaluator and dies with
            # queue.Full. Forking 256 subprocesses per correlation was the cost,
            # and it is also what triggers JAX's fork-deadlock warning.
            env(correlation, split="valid", num_envs=eval_num_envs, seed=seed + EVAL_SEED_OFFSET, asynchronous=False),
            n_episode_multiple=2,
            # Extra thinking time before acting is the test-time-compute knob
            # from the Sokoban work. Each extra value multiplies evaluation cost,
            # so it stays off until we actually analyse thinking time.
            steps_to_think=list(eval_steps_to_think),
        )
        for correlation in eval_correlations
    }
    out.total_timesteps = total_timesteps
    out.loss = dataclasses.replace(
        out.loss,
        # boxworld_drc33 inherits its loss from sokoban_resnet59, whose gamma
        # suits Sokoban's *dense* reward: a box push pays off within a few
        # steps. Our reward is terminal-only, and gamma=0.97 discounts a goal 83
        # steps away (the 90th percentile for 25x25 mazes) to 0.08 - effectively
        # invisible, while the step penalty is immediate. The discount horizon
        # must exceed the episode limit for the goal to be learnable at all.
        gamma=gamma,
        # Also inherited from the ResNet config; far.ai's own DRC setting is
        # 0.97, which trades a little variance for much less bias.
        vtrace_lambda=0.97,
    )
    # sokoban_resnet59 clips gradients at 2.5e-4, calibrated for a two-billion
    # step run. At our budget that is far too conservative to move the network;
    # 0.015 is the value far.ai use for DRC on Sokoban.
    out.max_grad_norm = max_grad_norm
    return out


RESNET_CHANNELS: tuple[int, ...] = (32,) * 9
RESNET_KERNEL: int = 3
"""The ResNet sized against the DRC it stands in for.

cleanba's ``GuezResNetConfig`` is Guez et al.'s own non-recurrent baseline for
the DRC, and ``sokoban_resnet59`` - the config every maze preset inherits from
- already carries it before ``boxworld_drc33`` swaps in the ConvLSTM. Using it
unmodified keeps the comparison to *one* change: the network.

Its default shape (32-64 channels, 4x4 kernels, nine stages) was sized for
80x80 Sokoban pixels and comes to 3.4M parameters on an 11x11 symbolic input.
DRC(3,3) on the same input is 1.45M, a million of which is the flattened
Dense(256) head both share, so what is being matched is the convolutional
trunk: nine stages of 32-channel 3x3 convolutions against the DRC's nine
3x3x32 ConvLSTM ticks per environment step. 1.23M parameters in total, same
channel width, same kernel, same count of sequential convolutions per decision,
and no state carried between steps - which is the variable.
"""


def maze_resnet(**kwargs) -> Args:
    """The maze preset with a non-recurrent ResNet in place of the DRC(3,3).

    Accepts exactly the arguments of :func:`maze_drc33`; every other setting -
    optimiser, loss, batch shape, evaluation arms, checkpoint schedule - is the
    DRC's, so a difference between the runs is attributable to the network.
    """
    out = maze_drc33(**kwargs)
    out.net = GuezResNetConfig(
        yang_init=False,
        norm=IdentityNorm(),
        normalize_input=False,
        channels=RESNET_CHANNELS,
        kernel_sizes=(RESNET_KERNEL,) * len(RESNET_CHANNELS),
        strides=(1,) * len(RESNET_CHANNELS),
    )
    return out


def maze_transformer(**kwargs) -> Args:
    """The maze preset with a non-recurrent ViT-style transformer for the DRC(3,3).

    Same contract as :func:`maze_resnet`. The network itself is documented in
    :mod:`goalmisgen.nets.transformer`; its defaults are the sizes used here.
    """
    out = maze_drc33(**kwargs)
    out.net = TransformerSpec(yang_init=False, norm=IdentityNorm(), normalize_input=False)
    return out


def preset_for(net: str):
    """Look a preset up by the short name the training script takes."""
    try:
        return PRESETS[net]
    except KeyError:
        raise ValueError(f"unknown network {net!r}; choose from {sorted(PRESETS)}") from None


PRESETS = {"drc33": maze_drc33, "resnet": maze_resnet, "vit": maze_transformer}


def with_final_checkpoint(config: Args) -> Args:
    """Make sure the weights a run ends on are actually written to disk.

    Checkpoints are saved when ``learner_policy_version`` lands in
    ``eval_at_steps``, and that set is inherited from ``boxworld_drc33``, whose
    entries were chosen for a two-billion-step budget. At our budgets it is
    sparse near the end: an 80M-step run stops at update 15,625 while the set
    jumps from 13,692 to 19,560, so the last 10M steps trained an agent that was
    then thrown away.

    Nothing downstream noticed, because taking the newest checkpoint still gives
    a coherent agent — just not the one the run ended on. Adding the final update
    costs one more save and removes the surprise.
    """
    batch = int(config.local_num_envs * config.num_steps * config.num_actor_threads * len(config.actor_device_ids))
    config.eval_at_steps = frozenset(config.eval_at_steps | {config.total_timesteps // batch})
    return config


def maze_smoke_test() -> Args:
    """Tiny configuration that trains on CPU in seconds.

    Exists to prove the environment, cleanba and JAX fit together end to end
    before any GPU time is spent. Not a scientific configuration.

    400k steps is 1250 updates. At 40k it was 125, which is too few for a
    "return improved" assertion to mean anything: the signal is noisy at that
    scale and the test passed or failed on luck. On a GPU this costs seconds.
    """
    out = maze_drc33(
        min_size=5,
        max_size=5,
        step_penalty=0.15,
        max_episode_steps=30,
        total_timesteps=400_000,
        eval_correlations=(1.0,),
    )
    out.train_env = dataclasses.replace(out.train_env, num_envs=16, asynchronous=False)
    out.eval_envs = {}

    out.local_num_envs = 16
    out.num_steps = 20
    out.num_minibatches = 1
    out.num_actor_threads = 1
    out.log_frequency = 1
    out.save_model = False
    out.eval_at_steps = frozenset()
    out.learning_rate = 4e-3
    out.final_learning_rate = 4e-4
    return out
