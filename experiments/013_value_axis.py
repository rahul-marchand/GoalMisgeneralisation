"""Fine-tune onto a new objective value, and keep the weight change.

    uv run python experiments/013_value_axis.py CHECKPOINT --value 0.7 \
        --levels /workspace/data/valueaxis/levels/v070 \
        --run-dir /workspace/data/valueaxis/runs/v070

``novalue11`` never saw a value channel. Its objectives were worth the constants
(1.0, 0.5), bound to colour, every single episode. So there is no per-episode
value for a probe to correlate against, and the whole correlational toolkit is
unavailable here by construction — which is the point. An agent trained this way
had no reason to represent "1.0" and "0.5" at all: it could have compiled the
comparison down into a threshold on the difference in distances and thrown the
values away.

Fine-tuning manufactures the variance that training withheld. Hold colour 0 at
1.0, retrain onto a new value ``v`` for colour 1, and the resulting change in
parameters is one sample of the map from value to weights. This script takes one
such sample; sweeping ``v`` and comparing the samples is what asks the real
question, which is not *where* the change lands but what *shape* the map has:

* if the diffs at different ``v`` are collinear, and scale with ``v``, then the
  agent holds what an objective is worth in something that behaves like a slot,
  and a direction in weight space can be fitted and written to
* if they are unrelated directions, each fine-tune rebuilt a threshold from
  scratch and there is no value quantity in there to find

Sparsity is deliberately not the criterion. An L0 or L1 penalty returns a sparse
diff whichever is true, so sparsity would measure the regulariser. Collinearity
across a grid cannot be produced that way.

Every arm must differ **only** in what the objectives pay: same seed, same
learning rate, same number of updates, and levels that are identical layouts
carrying different values (``FixedValues`` consumes no randomness, so a dataset
sweep generated at one seed has one set of mazes).

Two things about resuming are not obvious, and both are handled by
``reset_checkpoint`` below rather than by touching the training stack.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cleanba.cleanba_impala
import farconf
from cleanba.cleanba_impala import WandbWriter, train
from cleanba.config import Args

from goalmisgen.configs.presets import maze_drc33
from goalmisgen.configs.writers import CsvWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="The trained agent to fine-tune from.")
    parser.add_argument("--value", type=float, required=True, help="What colour 1 is worth.")
    parser.add_argument(
        "--value-zero",
        type=float,
        default=1.0,
        help="What colour 0 is worth. Sweeping this instead separates a value from a "
        "threshold: only the gap between the two drives behaviour, so a knob that is "
        "really the gap moves the same way whichever colour is changed, while a pair "
        "of value slots does not.",
    )
    parser.add_argument(
        "--objective-values",
        type=float,
        nargs="+",
        default=None,
        help="All objectives' values at once, overriding --value and --value-zero. Needed "
        "beyond two objectives, where the choice no longer reduces to a single difference.",
    )
    parser.add_argument("--levels", type=str, required=True, help="Dataset generated at this objective value.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Full run directory, not a parent of one.")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Held constant. Annealing would make the diffs at different values incomparable, "
        "since each arm would sit at a different point on its own schedule.",
    )
    parser.add_argument("--checkpoints", type=int, default=4, help="Evenly spaced, the last at the final update.")
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def finetune_config(args: argparse.Namespace) -> Args:
    """The base run's configuration, with the value swapped and the schedule flattened."""
    values = tuple(args.objective_values) if args.objective_values else (args.value_zero, args.value)
    config = maze_drc33(
        feature_value_correlation=1.0,
        min_size=args.size,
        max_size=args.size,
        n_objectives=len(values),
        objective_values=values,
        hide_values=True,
        total_timesteps=args.steps,
        level_dataset=args.levels,
        seed=args.seed,
    )

    # A fine-tune is short enough that an annealed rate would spend most of it
    # near zero, and each arm would anneal over its own run rather than sharing
    # one schedule. A constant rate is what makes the arms comparable.
    config.learning_rate = args.lr
    config.final_learning_rate = args.lr
    config.anneal_lr = False

    # Evaluation exists to track misgeneralisation across correlations during a
    # long run. Here the measurement is the weight diff, and the behavioural
    # readout is 006 on the finished checkpoint, so evaluation would only cost
    # rollout time. Dropping the environments also drops the one upstream path
    # that assumes Sokoban.
    config.eval_envs = {}
    config.eval_at_steps = checkpoint_updates(config, args.checkpoints)
    config.save_model = True
    return config


def local_batch_size(config: Args) -> int:
    return int(config.local_num_envs * config.num_steps * config.num_actor_threads * len(config.actor_device_ids))


def checkpoint_updates(config: Args, count: int) -> frozenset[int]:
    """Updates at which to save, the last being the final one.

    Checkpoints are written when ``learner_policy_version`` lands in this set, so
    the final update has to be a member or the run finishes with nothing saved.
    The intermediate ones are what let the analysis ask whether the diff's
    *direction* is settled long before its magnitude is.
    """
    total = config.total_timesteps // local_batch_size(config)
    return frozenset(total * i // count for i in range(1, count + 1))


def reset_checkpoint(base: Path, destination: Path, config: Args) -> Path:
    """A copy of ``base`` that ``train`` will resume from as though it were new.

    Two counters in the checkpoint would otherwise end the run before it starts.
    ``update_step`` is where the loop begins and ``learner_policy_version`` is
    where it stops, and the base checkpoint carries both from the end of a
    140M-step run — so a short fine-tune would iterate over an empty range and
    exit having done nothing. Both are reset here.

    The optimiser is also rebuilt from *this* file's config rather than from the
    caller's, which is how the constant learning rate actually takes effect.

    The ``model`` file is copied byte for byte instead of being re-serialised:
    loading sums the ConvLSTM fence kernel along its input axis, so a checkpoint
    written back out after a load no longer has the shape ``init_params`` builds,
    and the next load fails on it.
    """
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / "model", destination / "model")

    config = dataclasses.replace(config, learner_policy_version=0)
    payload = {"cfg": farconf.to_dict(config, Args), "update_step": 1}
    (destination / "cfg.json").write_text(json.dumps(payload, indent=2))
    return destination


def main() -> None:
    args = parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit or 'unknown'}\nargv   {' '.join(sys.argv[1:])}")

    config = finetune_config(args)
    updates = config.total_timesteps // local_batch_size(config)

    print(f"\nbase            {args.checkpoint}")
    print(f"objective values{config.train_env.objective_values}")
    print(f"levels          {args.levels}")
    print(f"steps           {config.total_timesteps:,}  ({updates:,} updates)")
    print(f"learning rate   {config.learning_rate:g}  constant")
    print(f"saving at       {sorted(config.eval_at_steps)}")
    print(f"run dir         {args.run_dir}")

    # The task's own exchange rate, which the agent is not told and has to be
    # driven to by reward alone. Printed here because it is what 006 measures
    # afterwards, and having it on the run's own log makes the two comparable
    # without going back to the config.
    values = config.train_env.objective_values
    best, second = sorted(values, reverse=True)[:2]
    print(
        f"\ntop two objectives worth {best:g} and {second:g}, so the richer is worth "
        f"{(best - second) / config.train_env.step_penalty:.1f} extra steps\n"
    )

    config.load_path = reset_checkpoint(args.checkpoint, args.run_dir / "init", config)
    config.base_run_dir = args.run_dir

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
    print("FINETUNE COMPLETE")


if __name__ == "__main__":
    main()
