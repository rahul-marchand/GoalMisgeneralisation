"""What one cell of the scaling grid actually costs: peak memory and steps per second.

    uv run python scripts/shape_probe.py --all --batch 1024
    uv run python scripts/shape_probe.py --d-model 512 --layers 16 --batch 1024 --no-remat

Everything in ``Preregistration-scaling.md`` about what the campaign costs rests
on two estimates: that ``nn.remat`` takes the largest cell's activations from
about 161 GB to about 16 GB, and that a step is fast enough for the nine bases
to finish in a few hours. Both are arithmetic until a card has run them. This
runs them, on synthetic data, so it needs no dataset and can be the first thing
executed on a fresh pod.

The measurement is the *training* step -- forward, backward, optimiser -- under
the same AdamW the campaign uses, because the optimiser state is a third of the
memory at the top of the grid and leaving it out would flatter the answer.

**One shape per process.** JAX's allocator reports a high-water mark and offers
no way to reset it, so measuring several shapes in one process reports the
largest of them for all of them. ``--all`` therefore re-runs this file per shape
rather than looping in-process; the first version of this script did loop, and
reported the 50M cell's peak for every cell including the 0.8M one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

GRID: tuple[tuple[int, int], ...] = (
    (128, 4),
    (256, 4),
    (512, 4),
    (128, 8),
    (256, 8),
    (512, 8),
    (128, 16),
    (256, 16),
    (512, 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Every cell of the grid, one subprocess each.")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        metavar="DxL",
        help="Override the grid, e.g. --shapes 512x16 256x16. Useful for re-probing only the cells that failed.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=12, help="Timed steps, after two warm-up steps.")
    parser.add_argument("--no-remat", dest="remat", action="store_false", help="Measure the cost of remat by leaving it off.")
    parser.add_argument("--total-steps", type=int, default=30_000, help="Base length, for the wall-clock estimate.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Precision the matmuls run in. Parameters stay float32 either way.",
    )
    parser.add_argument("--json", action="store_true", help="One JSON object on stdout, nothing else.")
    return parser.parse_args()


def probe(args: argparse.Namespace) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training import train_state

    from goalmisgen.offline.model import (
        ModelConfig,
        RoutePrefixLM,
        cross_entropy,
        parameter_count,
        targets_from_routes,
    )
    from goalmisgen.offline.train import initial_params

    config = ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.d_model // 32)
    model = RoutePrefixLM(config, dtype=getattr(jnp, args.dtype), remat=args.remat)
    params = initial_params(model, jax.random.PRNGKey(0))["params"]
    n_params = parameter_count(params)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params={"params": params},
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=0.01)),
    )

    rng = np.random.default_rng(0)
    observations = jnp.asarray(rng.random((args.batch, config.size, config.size, config.n_channels)), jnp.float32)
    actions = jnp.asarray(rng.integers(0, config.n_actions, (args.batch, config.max_actions)), jnp.int32)
    lengths = jnp.asarray(rng.integers(4, 20, args.batch), jnp.int32)

    @jax.jit
    def step(state, observations, actions, lengths):
        targets = targets_from_routes(actions, lengths, config.eos)

        def loss_fn(p):
            return cross_entropy(state.apply_fn(p, observations, actions)[0], targets)

        return state.apply_gradients(grads=jax.grad(loss_fn)(state.params))

    for _ in range(2):  # compile, and let the allocator settle
        state = step(state, observations, actions, lengths)
    jax.block_until_ready(state.params)

    started = time.perf_counter()
    for _ in range(args.steps):
        state = step(state, observations, actions, lengths)
    jax.block_until_ready(state.params)
    seconds = (time.perf_counter() - started) / args.steps

    # The useful FLOPs, excluding remat's extra forward pass, so that MFU across
    # a remat and a no-remat run answers "how well is the card being used" rather
    # than "was recomputation on".
    tokens = args.batch * config.sequence_length
    attention = 3 * 4 * config.sequence_length**2 * args.d_model * args.layers * args.batch
    flops = 6 * n_params * tokens + attention

    device = jax.local_devices()[0]
    try:
        peak = device.memory_stats().get("peak_bytes_in_use")
    except (AttributeError, TypeError):
        peak = None

    return {
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": config.d_model // 32,
        "parameters": n_params,
        "batch": args.batch,
        "remat": args.remat,
        "dtype": args.dtype,
        "platform": device.platform,
        "device": device.device_kind,
        "seconds_per_step": seconds,
        "steps_per_second": 1 / seconds,
        "peak_gb": None if peak is None else peak / 2**30,
        "tflops": flops / seconds / 1e12,
        "base_hours": seconds * args.total_steps / 3600,
    }


def main() -> None:
    args = parse_args()
    if not args.all:
        row = probe(args)
        print(json.dumps(row) if args.json else json.dumps(row, indent=2))
        return

    grid = GRID if not args.shapes else tuple(tuple(int(part) for part in s.split("x")) for s in args.shapes)
    rows = []
    for d_model, layers in grid:
        command = [
            sys.executable,
            __file__,
            "--json",
            "--d-model",
            str(d_model),
            "--layers",
            str(layers),
            "--batch",
            str(args.batch),
            "--steps",
            str(args.steps),
            "--total-steps",
            str(args.total_steps),
        ]
        if not args.remat:
            command.append("--no-remat")
        finished = subprocess.run(command, capture_output=True, text=True)
        if finished.returncode:
            tail = finished.stderr.strip().splitlines()[-1:] or ["(no output)"]
            print(f"  d{d_model:<4} L{layers:<3} FAILED: {tail[0][:120]}", flush=True)
            rows.append({"d_model": d_model, "layers": layers, "failed": tail[0]})
            continue
        row = json.loads(finished.stdout)
        rows.append(row)
        if len(rows) == 1:
            print(f"device {row['device']} ({row['platform']}), batch {row['batch']}, remat {row['remat']}, {row['dtype']}\n")
            print(f"  {'d':>5}{'L':>4}{'params':>14}{'ms/step':>10}{'peak GB':>10}{'TFLOP/s':>10}{'base h':>9}")
        peak = "-" if row["peak_gb"] is None else f"{row['peak_gb']:.1f}"
        print(
            f"  {row['d_model']:>5}{row['layers']:>4}{row['parameters']:>14,}"
            f"{row['seconds_per_step'] * 1000:>10.1f}{peak:>10}{row['tflops']:>10.1f}{row['base_hours']:>9.2f}",
            flush=True,
        )

    done = [r for r in rows if "failed" not in r]
    if done:
        print(f"\n  {len(done)} bases, sequential: {sum(r['base_hours'] for r in done):.1f} h on this card")
        print(f"  plus arms at roughly 0.4x: {1.4 * sum(r['base_hours'] for r in done):.1f} h")
    print("\n" + json.dumps(rows))


if __name__ == "__main__":
    main()
