"""Keep only the channels that carry the axis, and see what still works.

    uv run python experiments/019_restrict_the_axis.py \
        --base /workspace/data/runs/novalue11/local-files/cp_140206080 \
        --arms /workspace/data/valueaxis/runs --prefix v --base-value 0.5 \
        --levels /workspace/data/valueaxis/levels/v050 --objective-values 1.0 0.5

``018`` found the axis enriched about twice over in a couple of hidden channels
of each recurrent layer. This asks what happens if the rest is thrown away.

Two things could come of it, and they are worth keeping apart.

**Denoising.** A fitted axis is roughly a sixth signal, and the arm-specific part
of each fine-tune is spread over the whole network while the value-carrying part
is not. Masking to the enriched channels therefore discards more noise than
signal, and the estimate can *improve* — which would show up as the weight-space
tests that currently fail beginning to work. Reading an offset back off an arm is
the one that matters: it is the same question as recovering what an agent was
trained to want, from its weights.

**Sufficiency.** If writing only those channels still moves the exchange rate,
that is a localisation claim with a number attached — this fraction of the
network is enough. If it does not, the concentration is real but the rest of the
axis is doing the work, which is equally worth knowing.

Masked directions are written twice: as they are, which asks whether those
coordinates suffice at their own magnitude, and rescaled to the full axis's
length, which asks whether the direction inside them is the right one
irrespective of how much of it survived the mask.
"""

from __future__ import annotations

import argparse
import re
from functools import partial
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state
from jax.flatten_util import ravel_pytree

from goalmisgen import provenance
from goalmisgen.analysis import collect_episode_outcomes, summarise
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.weights import fit_axis_and_drift, projected_offset
from goalmisgen.configs.env import MazeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="v")
    parser.add_argument(
        "--rank-from",
        type=str,
        default=None,
        metavar="PREFIX",
        help="A second sweep, used only to choose which channels to keep. Choosing them "
        "from the sweep being tested would leak: the same arms would pick the mask and "
        "then be scored through it.",
    )
    parser.add_argument("--rank-base-value", type=float, default=None)
    parser.add_argument("--base-value", type=float, required=True)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--objective-values", type=float, nargs="+", required=True)
    parser.add_argument("--write-offset", type=float, default=0.4, help="Offset to write when testing behaviour.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--at", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-behaviour", action="store_true")
    return parser.parse_args()


def named_leaves(params) -> list[tuple[str, np.ndarray]]:
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    return [("/".join(str(k.key) for k in path), np.asarray(v)) for path, v in flat]


def channel_masks(shapes: dict[str, tuple], keep: dict[int, np.ndarray], encoder: bool = False) -> dict[str, np.ndarray]:
    """One boolean array per parameter, keeping chosen channels of each recurrent layer.

    A gate convolution's last axis is ``4 x hidden``, so hidden channel ``c``
    owns indices ``c, 32+c, 64+c, 96+c`` — one per gate. Everything outside the
    recurrent cells is dropped entirely, which is the point: the question is
    whether those channels alone carry the axis.
    """
    masks = {}
    for name, shape in shapes.items():
        mask = np.zeros(shape, dtype=bool)
        if encoder and "network_params" in name and "cell_list" not in name:
            mask[...] = True
        cell = re.search(r"cell_list_(\d+)/(ih|hh)/kernel", name)
        if cell is not None and int(cell.group(1)) in keep:
            hidden = shape[-1] // 4
            for gate in range(4):
                for channel in keep[int(cell.group(1))]:
                    mask[..., gate * hidden + channel] = True
        masks[name] = mask
    return masks


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")

    config = MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=args.size,
        max_size=args.size,
        n_objectives=len(args.objective_values),
        objective_values=tuple(args.objective_values),
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )
    policy, _, _, base_state, _ = load_train_state(args.base, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    names = [name for name, _ in named_leaves(base_state.params)]
    shapes = {name: value.shape for name, value in named_leaves(base_state.params)}
    sizes = [value.size for _, value in named_leaves(base_state.params)]
    bounds = np.cumsum(sizes)[:-1]

    offsets, stack = [], []
    for run in sorted(args.arms.iterdir()):
        match = re.fullmatch(rf"{args.prefix}(\d{{3}})", run.name)
        if not match or not (run / "local-files").is_dir():
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            continue
        offset = int(match.group(1)) / 100 - args.base_value
        if abs(offset) < 1e-9:
            continue
        _, _, _, state, _ = load_train_state(checkpoints[args.at], env_cfg=config)
        offsets.append(offset)
        stack.append(np.asarray(ravel_pytree(state.params)[0] - base_flat, dtype=np.float64))
    offsets, stack = np.array(offsets), np.stack(stack)
    print(f"{len(offsets)} arms at offsets {sorted(round(float(o), 2) for o in offsets)}\n")

    axis, drift = fit_axis_and_drift(offsets, stack)

    # Rank each cell's channels by how much more of the axis they carry than of
    # the shared component, exactly as 018 does.
    def per_channel(vector, name):
        piece = dict(zip(names, np.split(vector, bounds)))[name].reshape(shapes[name])
        per_output = (piece**2).sum(axis=tuple(range(piece.ndim - 1)))
        return per_output.reshape(4, -1).sum(axis=0)

    ranking_axis, ranking_drift, ranked_by = axis, drift, f"{args.prefix} (the sweep under test — leaks)"
    if args.rank_from:
        other_offsets, other_stack = [], []
        for run in sorted(args.arms.iterdir()):
            match = re.fullmatch(rf"{args.rank_from}(\d{{3}})", run.name)
            if not match or not (run / "local-files").is_dir():
                continue
            checkpoints = sorted((run / "local-files").glob("cp_*"))
            if not checkpoints:
                continue
            offset = int(match.group(1)) / 100 - (args.rank_base_value if args.rank_base_value is not None else args.base_value)
            if abs(offset) < 1e-9:
                continue
            _, _, _, state, _ = load_train_state(checkpoints[args.at], env_cfg=config)
            other_offsets.append(offset)
            other_stack.append(np.asarray(ravel_pytree(state.params)[0] - base_flat, dtype=np.float64))
        ranking_axis, ranking_drift = fit_axis_and_drift(np.array(other_offsets), np.stack(other_stack))
        ranked_by = f"{args.rank_from} ({len(other_offsets)} arms, independent of the sweep under test)"

    ranking = {}
    for cell in range(3):
        name = f"params/network_params/cell_list_{cell}/ih/kernel"
        if name not in shapes:
            continue
        a, d = per_channel(ranking_axis, name), per_channel(ranking_drift, name)
        ranking[cell] = np.argsort((a / a.sum()) / (d / d.sum()))[::-1]
    print(f"channels ranked using sweep {ranked_by}, best first:")
    for cell, order in ranking.items():
        print(f"  cell_list_{cell}: " + " ".join(f"ch{c:02d}" for c in order[:8]))

    variants: dict[str, np.ndarray] = {"full axis": np.ones_like(axis, dtype=bool)}
    cells_only = {cell: order for cell, order in ranking.items()}
    for label, keep, encoder in (
        ("all layers, all channels", {c: o for c, o in cells_only.items()}, False),
        ("all layers, top 2 channels", {c: o[:2] for c, o in cells_only.items()}, False),
        ("layer 0 only, top 2 channels", {0: cells_only[0][:2]}, False),
        ("layer 1 only, top 2 channels", {1: cells_only[1][:2]}, False),
        ("layer 2 only, top 2 channels", {2: cells_only[2][:2]}, False),
        ("encoder only", {}, True),
        ("encoder + layer 0 top 2", {0: cells_only[0][:2]}, True),
        ("encoder + all layers top 2", {c: o[:2] for c, o in cells_only.items()}, True),
    ):
        masks = channel_masks(shapes, keep, encoder)
        variants[label] = np.concatenate([masks[name].ravel() for name in names])

    print("\n\n=== can an offset be read back off an arm it never saw? ===\n")
    print(f"  {'kept':>28}{'params':>10}{'of axis':>9}   implied offsets against the true ones")
    truth = "  ".join(f"{o:+.2f}" for o in offsets)
    print(f"  {'':>28}{'':>10}{'':>9}   {truth}   <- true")
    for label, mask in variants.items():
        share = float(np.sum(np.where(mask, axis, 0.0) ** 2) / np.sum(axis**2))
        implied = []
        for held in range(len(offsets)):
            others = [i for i in range(len(offsets)) if i != held]
            axis_h, drift_h = fit_axis_and_drift(offsets[others], stack[others])
            implied.append(projected_offset(np.where(mask, stack[held] - drift_h, 0.0), np.where(mask, axis_h, 0.0)))
        implied = np.array(implied)
        slope = float(np.polyfit(offsets, implied, 1)[0])
        correlation = float(np.corrcoef(offsets, implied)[0, 1])
        print(
            f"  {label:>28}{int(mask.sum()):>10,}{share:>9.1%}   "
            + "  ".join(f"{v:+.2f}" for v in implied)
            + f"   r {correlation:+.2f}  slope {slope:+.2f}"
        )
    print(
        "\n  Every axis here is fitted without the arm it is scoring. Done in-sample this\n"
        "  table reproduces the true offsets exactly and means nothing: the axis is a\n"
        "  weighted sum of these same arms, each arm's weight is proportional to its own\n"
        "  offset, and its own noise dominates its own projection. So it recovers what\n"
        "  the estimator put there. Held out, that route is closed.\n\n"
        "  A slope of 1 would mean the offset can be read straight off a checkpoint the\n"
        "  axis never saw. A slope near 0 with the same number returned for every arm is\n"
        "  the failure 014 reported."
    )

    if args.skip_behaviour:
        return

    print(f"\n\n=== does writing only those channels still move behaviour? ===\n")
    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    def measure(params, label):
        carry = policy.apply(params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
        state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

        def act(observations, starts):
            state["carry"], action, _, state["key"] = get_action(
                params, state["carry"], observations, starts, state["key"], temperature=0.0
            )
            return np.asarray(action)

        outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
        gaps, took, _ = value_distance_decisions(outcomes)
        point = indifference_point(gaps, took)
        print(f"  {label:>44}{point:>8.1f}   reached {summarise(outcomes).reached_objective:>6.1%}")
        return point

    print(f"  {'':>44}{'steps':>8}")
    measure(base_state.params, "base, untouched")
    for label, mask in variants.items():
        masked = np.where(mask, axis, 0.0)
        measure(unravel(base_flat + args.write_offset * masked), f"{label}, as is")
        if label != "full axis":
            scale = float(np.linalg.norm(axis) / np.linalg.norm(masked))
            measure(unravel(base_flat + args.write_offset * scale * masked), f"{label}, rescaled x{scale:.1f}")
    print(
        f"\n  Written at offset {args.write_offset:+.2f}, where the full axis reaches about 2.4 steps\n"
        "  and the arm trained there reaches 1.6, against the base's 7.7. 'As is' asks\n"
        "  whether these coordinates suffice at their own size; 'rescaled' asks whether\n"
        "  the direction inside them is right once the lost magnitude is put back."
    )


if __name__ == "__main__":
    main()
