"""Train an agent on multi-objective mazes. A DRC(3,3) unless told otherwise.

The first real training run. Evaluation environments differ from training only
in the value/colour correlation and in drawing from the held-out split, so the
gap between the ``rho100`` and ``rho000`` curves *is* the goal misgeneralisation
measurement, tracked throughout training rather than only at the end.

``--net`` swaps the policy network and nothing else: ``resnet`` is cleanba's
own non-recurrent ResNet, ``vit`` a transformer of ours. Both presets are the
DRC's to the last hyperparameter except the network, so a difference between
the runs is the architecture's.

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
from goalmisgen.configs.presets import PRESETS, preset_for, with_final_checkpoint
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
    parser.add_argument(
        "--net",
        type=str,
        default="drc33",
        choices=sorted(PRESETS),
        help="Which policy network. Everything else in the preset is identical across choices.",
    )
    parser.add_argument("--correlation", type=float, default=None, help="Training rho.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--step-penalty", type=float, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-objectives", type=int, default=None)
    parser.add_argument(
        "--objective-values",
        type=float,
        nargs="+",
        default=None,
        help="What each objective is worth. Must match the level dataset, which stores them.",
    )
    parser.add_argument("--randomise-values", action="store_true")
    parser.add_argument(
        "--hide-values",
        action="store_true",
        help="Drop the value channel, so the agent must learn what each objective is worth "
        "instead of reading it. No misgeneralisation can be measured on such a run.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Why this run is being launched, written to NOTE.md beside its checkpoints. "
        "The configuration is saved anyway; the reason is not, and it is the part that "
        "cannot be reconstructed later from what is on disk.",
    )
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
            ("n_objectives", args.n_objectives),
            ("objective_values", tuple(args.objective_values) if args.objective_values else None),
        )
        if value is not None
    }
    if args.randomise_values:
        overrides["randomise_values"] = True
    if args.hide_values:
        overrides["hide_values"] = True
    config = with_final_checkpoint(preset_for(args.net)(**overrides))
    config.base_run_dir = args.run_dir
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.note:
        (args.run_dir / "NOTE.md").write_text(args.note.strip() + "\n")

    evaluated_at = sorted(e.env.feature_value_correlation for e in config.eval_envs.values() if isinstance(e.env, MazeConfig))
    # Report the resolved configuration, not the raw flags: an unset flag leaves
    # the preset in charge, so printing the flag would say "None" for a run that
    # is in fact training at the preset's correlation.
    print(f"network        {args.net}  ({type(config.net).__name__})")
    print(f"training rho   {config.train_env.feature_value_correlation}")
    print(f"evaluating rho {evaluated_at}")
    print(f"maze size      {config.train_env.min_size}-{config.train_env.max_size}")
    print(f"objectives     {config.train_env.n_objectives} worth {config.train_env.objective_values}")
    print(f"step penalty   {config.train_env.step_penalty}   gamma {config.loss.gamma}")
    print(f"levels         {args.levels or 'sampled live'}")
    print(f"timesteps      {config.total_timesteps:,}")
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
