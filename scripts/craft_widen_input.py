"""Warm-start a checkpoint onto an observation with one more channel.

    uv run python scripts/craft_widen_input.py RUN_DIR NEW_RUN_DIR --insert-channel 4 [--checkpoint STEP]

The route model's only channel-shaped parameter is the cell embedding's
kernel, ``(n_channels, d_model)``; a zero row inserted at the new channel's
index leaves every observation the model saw unchanged and lets training
continue from where it was. Written when the crafting observation gained
its table channel after 60k steps had been spent without one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax.serialization
import numpy as np

from goalmisgen.offline.train import list_checkpoints, load_checkpoint, load_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path, help="A new run directory holding config.json and checkpoints/step_00000000.")
    parser.add_argument("--insert-channel", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, default=None)
    args = parser.parse_args()
    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.checkpoint is None else next(c for c in checkpoints if c[0] == args.checkpoint)
    model, params = load_checkpoint(directory)
    kernel = np.asarray(params["params"]["cell_in"]["kernel"])
    assert kernel.shape[0] == model.config.n_channels, kernel.shape
    widened = np.insert(kernel, args.insert_channel, 0.0, axis=0)
    params["params"]["cell_in"]["kernel"] = widened
    config = load_run_config(args.run)
    config["model"]["n_channels"] = model.config.n_channels + 1
    config["widened_from"] = {"run": str(args.run), "step": step, "inserted_channel": args.insert_channel}
    out = args.out / "checkpoints" / "step_00000000"
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{args.run.name} @ {step}: cell_in kernel {kernel.shape} -> {widened.shape}; wrote {out}")


if __name__ == "__main__":
    main()
