"""Train a DRC(3,3) on multi-objective mazes.

The first real training run. Evaluation environments differ from training only
in the value/colour correlation and in drawing from the held-out split, so the
gap between the ``rho100`` and ``rho000`` curves *is* the goal misgeneralisation
measurement, tracked throughout training rather than only at the end.

    # short profiling run, to size things before committing
    uv run python experiments/001_maze_repro.py --total-timesteps 2000000 \
        --levels /workspace/data/levels

    # full run
    uv run python experiments/001_maze_repro.py --levels /workspace/data/levels

Watch steps/second alongside GPU and CPU utilisation. If the GPU is idle while
the CPUs are pegged, the bottleneck is the environment, not the model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cleanba.cleanba_impala
from cleanba.cleanba_impala import WandbWriter, train

from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33
from goalmisgen.configs.writers import CsvWriter


def parse_args() -> argparse.Namespace:
    """Every option defaults to None so an unset flag leaves the preset in charge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Pre-generated level directory. Omit to sample levels live, which is "
        "roughly 350x more expensive per reset and may starve the GPU.",
    )
    parser.add_argument("--correlation", type=float, default=None, help="Training rho.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--step-penalty", type=float, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/workspace/data/runs"),
        help="Where checkpoints go. Use the persistent volume so a reclaimed pod does not lose the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Only pass what was given on the command line. Passing every argparse
    # default would shadow the preset's own defaults, making them dead code.
    overrides = {
        name: value
        for name, value in (
            ("feature_value_correlation", args.correlation),
            ("min_size", args.min_size),
            ("max_size", args.max_size),
            ("step_penalty", args.step_penalty),
            ("max_episode_steps", args.max_episode_steps),
            ("total_timesteps", args.total_timesteps),
            ("seed", args.seed),
            ("level_dataset", args.levels),
        )
        if value is not None
    }
    if args.randomise_values:
        overrides["randomise_values"] = True
    config = maze_drc33(**overrides)
    config.base_run_dir = args.run_dir

    evaluated_at = sorted(e.env.feature_value_correlation for e in config.eval_envs.values() if isinstance(e.env, MazeConfig))
    print(f"training rho   {args.correlation}")
    print(f"evaluating rho {evaluated_at}")
    print(f"levels         {args.levels or 'sampled live'}")
    print(f"timesteps      {args.total_timesteps:,}")
    print(f"run dir        {args.run_dir}\n")

    # Weights & Biases needs credentials the machine may not have (a fresh cloud
    # pod, CI). Fall back to a local CSV rather than failing at startup.
    if os.environ.get("WANDB_MODE") in ("disabled", "offline") or not os.environ.get("WANDB_API_KEY"):
        print("W&B unavailable; logging metrics to CSV instead\n")
        writer = CsvWriter(config, args.run_dir)
    else:
        writer = WandbWriter(config)

    cleanba.cleanba_impala.MUST_STOP_PROGRAM = False
    try:
        train(config, writer=writer)
    finally:
        if isinstance(writer, CsvWriter):
            writer.flush()
    print("RUN COMPLETE")


if __name__ == "__main__":
    main()
