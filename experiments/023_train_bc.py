"""Train a route transformer on expert demonstrations, scoring it as it goes.

    uv run python experiments/023_train_bc.py \\
        --demos /workspace/data/offline/demos/train.rho100 \\
        --eval rho100=/workspace/data/offline/demos/valid.rho100 \\
               rho050=/workspace/data/offline/demos/valid.rho050 \\
               rho000=/workspace/data/offline/demos/valid.rho000 \\
        --out /workspace/data/offline/runs/bc11.rho100.s1 --seed 1

The offline twin of ``001_maze_repro.py``. The demonstrations carry the
colour-value correlation; nothing else about the training signal knows that
colour exists. At every checkpoint the model's greedy routes on held-out levels
are replayed under the environment's rules at each evaluation correlation, so
the run writes the same ``chose_optimal`` / ``followed_feature_zero`` curves
across rho that ``002_measure_proxy.py`` reads off a DRC agent - and writes
them from the first steps, where the early-warning question lives.

What would refute the hypothesis that imitation of correlated data produces
goal misgeneralisation: a model trained at rho=1.0 whose ``chose_optimal`` at
rho=0.0 matches its rho=1.0 figure, i.e. one that learned the value-distance
trade rather than the colour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.train import TrainConfig, load_run_config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", type=Path, required=True, help="Training demonstrations.")
    parser.add_argument(
        "--eval",
        type=str,
        nargs="*",
        default=[],
        help="name=path pairs of held-out demonstration sets to decode at every checkpoint.",
    )
    parser.add_argument("--eval-levels", type=int, default=1024, help="Levels decoded per evaluation set.")
    parser.add_argument("--out", type=Path, required=True, help="Run directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--checkpoint-first", type=int, default=25)
    parser.add_argument("--checkpoint-ratio", type=float, default=1.4)
    parser.add_argument(
        "--hide-values",
        action="store_true",
        help="Drop the value channel from every observation (training and evaluation), so the "
        "values are learned constants - the twin of novalue11's colour_is_the_only_value_cue.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Checkpoint directory to fine-tune from, instead of a fresh initialisation. The "
        "model shape and --hide-values are taken from that run's config.json.",
    )
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--note", type=str, default=None, help="Why this run exists; written beside the run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hide_values = args.hide_values
    if args.init_from is not None:
        source = load_run_config(args.init_from.parent.parent)
        hide_values = bool(source["demos"].get("hide_values", False))
        model_config = ModelConfig.from_dict(source["model"])
    demos = DemoSet.load(args.demos, hide_values=hide_values)
    if args.init_from is None:
        model_config = ModelConfig(
            size=demos.size,
            n_channels=demos.n_channels,
            max_actions=demos.max_actions,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
        )
    elif model_config.n_channels != demos.n_channels:
        raise SystemExit(f"{args.init_from} reads {model_config.n_channels} channels but {args.demos} gives {demos.n_channels}")
    train_config = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        seed=args.seed,
        checkpoint_first=args.checkpoint_first,
        checkpoint_ratio=args.checkpoint_ratio,
        schedule=args.schedule,
        init_from=None if args.init_from is None else str(args.init_from),
    )

    held_out = {}
    for item in args.eval:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--eval expects name=path, got {item!r}")
        held_out[name] = DemoSet.load(path, hide_values=hide_values)
    for name, other in held_out.items():
        shared = np.intersect1d(np.asarray(demos.level_index), np.asarray(other.level_index))
        if len(shared) and other.meta.get("source_fingerprint") == demos.meta.get("source_fingerprint"):
            raise SystemExit(f"evaluation set {name} shares {len(shared)} levels with the training set")
    indices = np.arange(min(args.eval_levels, *(len(d) for d in held_out.values()))) if held_out else None

    args.out.mkdir(parents=True, exist_ok=True)
    if args.note:
        (args.out / "note.txt").write_text(args.note.strip() + "\n")
    print(f"training on {args.demos} (rho={demos.rho}, {len(demos):,} demonstrations, values {'hidden' if hide_values else 'shown'})")
    if args.init_from is not None:
        print(f"fine-tuning from {args.init_from}")
    print(f"model {model_config}")
    print(f"train {train_config}")
    if held_out:
        print(
            "evaluating on "
            + ", ".join(f"{name} (rho={d.rho})" for name, d in held_out.items())
            + f", {len(indices)} levels each"
        )
    print()

    model = RoutePrefixLM(model_config)

    def evaluator(params, step: int) -> dict[str, float]:
        row: dict[str, float] = {}
        parts = []
        for name, other in held_out.items():
            summary, _, _ = evaluate(model, params, other, indices)
            for key, value in summary.as_row().items():
                row[f"{name}/{key}"] = value
            parts.append(
                f"{name}: reached {summary.behaviour.reached_objective:.2f} optimal "
                f"{summary.behaviour.chose_optimal:.2f} f0 {summary.behaviour.followed_feature_zero:.2f} "
                f"legal {summary.legal:.2f}"
            )
        print(f"[checkpoint {step:>7,}] " + " | ".join(parts), flush=True)
        return row

    train(
        demos,
        model_config,
        train_config,
        args.out,
        evaluate=evaluator if held_out else None,
        log=lambda s: print(s, flush=True),
    )
    (args.out / "done.json").write_text(json.dumps({"steps": args.steps}))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
