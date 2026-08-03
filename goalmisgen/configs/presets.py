"""Training presets for maze experiments.

``maze_drc33`` inherits far.ai's DRC(3,3) network, IMPALA loss and optimiser
settings from ``boxworld_drc33`` — their tuned configuration for a pure-Python
environment — and swaps in the maze. Only the environment differs, so any
difference in results is attributable to the task rather than to hyperparameters
we invented.
"""

from __future__ import annotations

import dataclasses

from cleanba.config import Args, boxworld_drc33
from cleanba.evaluate import EvalConfig

from goalmisgen.configs.env import MazeConfig

DEFAULT_EVAL_CORRELATIONS: tuple[float, ...] = (1.0, 0.5, 0.0)
"""Evaluate on the training correlation, an uninformative one, and a reversed one.

Tracking all three throughout training shows *when* a proxy is adopted, not just
whether it was present at the end.
"""


def maze_drc33(
    feature_value_correlation: float = 1.0,
    eval_correlations: tuple[float, ...] = DEFAULT_EVAL_CORRELATIONS,
    min_size: int = 5,
    max_size: int = 25,
    n_objectives: int = 2,
    step_penalty: float = 0.05,
    randomise_values: bool = False,
    max_episode_steps: int = 120,
    total_timesteps: int = 200_000_000,
    seed: int = 1234,
    level_dataset: str | None = None,
) -> Args:
    """DRC(3,3) on multi-objective mazes.

    Evaluation environments differ from training in ``feature_value_correlation``
    — the misgeneralisation measurement — and, when a level dataset is used, in
    which split they draw from. Evaluating on training levels would confound
    misgeneralisation with memorisation, so the two must never share levels.
    """
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
            feature_value_correlation=correlation,
            level_dataset=level_dataset,
            dataset_split=split,
            seed=seed,
        )
        settings.update(overrides)
        return MazeConfig(**settings)  # type: ignore[arg-type]

    out.train_env = env(feature_value_correlation, split="train")
    out.eval_envs = {
        f"rho{int(round(correlation * 100)):03d}": EvalConfig(
            env(correlation, split="valid", num_envs=256, seed=seed + 1),
            n_episode_multiple=4,
            # Extra thinking time before acting: the test-time-compute knob that
            # revealed iterative plan refinement in the Sokoban work.
            steps_to_think=[0, 2, 4, 8],
        )
        for correlation in eval_correlations
    }
    out.total_timesteps = total_timesteps
    return out


def maze_smoke_test() -> Args:
    """Tiny configuration that trains on CPU in seconds.

    Exists to prove the environment, cleanba and JAX fit together end to end
    before any GPU time is spent. Not a scientific configuration.
    """
    out = maze_drc33(
        min_size=5,
        max_size=5,
        step_penalty=0.15,
        max_episode_steps=30,
        total_timesteps=40_000,
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
