"""Write the value axis one parameter block at a time.

    uv run python scripts/decode_masked.py RUN --out DIR --n 20000

Part 3 found the axis multiplies the threshold, log-linearly in the offset over
its working range. The candidate explanation is compounding: components in
several layers that each scale the signal give an end-to-end gain of
``prod(1 + c_l * offset) ~ exp(c * offset)``. That predicts the ``log theta``
shifts of per-block writes should add up to the full write's shift, and it puts
the deep-write zero crossing at ``-1/max(c_l)``.

Blocks are the transformer's own top-level parameter groups: ``block_0`` to
``block_3``, the embedding side (cell/row/col/action) as ``embed``, and
``ln_final`` plus ``head`` as ``head``. The masks partition the flat parameter
vector exactly; that is asserted rather than assumed.

Output matches ``scripts/decode_grid.py``, one ``.npz`` per (block, offset).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import numpy as np

from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
from goalmisgen.offline.decode import replay_all
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.fast_decode import greedy_decode_cached
from goalmisgen.volume import offset_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Base run directory, e.g. .../runs/bcnv11.s1")
    parser.add_argument("--sweep", default="o0")
    parser.add_argument("--steps", type=int, default=1000, help="Arm budget, part of the arm directory name.")
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.45, -0.20, 0.20, 0.45])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--demos", default="/workspace/data/offline/demos/test.rho100")
    return parser.parse_args()


def group_of(path) -> str:
    """The block a parameter belongs to, from its tree path."""
    name = str(getattr(path[1], "key", path[1]))
    if name.startswith("block_"):
        return name
    if name in ("ln_final", "head"):
        return "head"
    return "embed"


def block_masks(params, n_total: int) -> dict[str, np.ndarray]:
    """Boolean masks over the raveled parameter vector, one per block.

    ``ravel_pytree`` concatenates leaves in tree order, which is the same order
    ``tree_flatten_with_path`` walks, so sizes line up positionally.
    """
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    masks: dict[str, np.ndarray] = {}
    start = 0
    for path, leaf in leaves:
        size = int(np.prod(leaf.shape)) if leaf.shape else 1
        group = group_of(path)
        masks.setdefault(group, np.zeros(n_total, dtype=bool))[start : start + size] = True
        start += size
    assert start == n_total, (start, n_total)
    total = np.zeros(n_total, dtype=int)
    for mask in masks.values():
        total += mask
    assert (total == 1).all(), "blocks must partition the parameter vector"
    return masks


def main() -> None:
    args = parse_args()
    base = load_base(args.run)
    arms = arm_dirs(args.run, args.sweep, args.steps)
    diffs = load_diffs(base, arms)
    offsets = np.array(sorted(o for o in diffs if abs(o) > 1e-9))
    axis, _ = fit_axis_and_drift(offsets, np.stack([diffs[o] for o in offsets]))
    masks = block_masks(base.params, len(base.flat))
    share = {g: float(np.linalg.norm(axis[m]) ** 2) for g, m in masks.items()}
    norm2 = float(np.linalg.norm(axis) ** 2)
    print(f"axis from {len(offsets)} arms; squared-norm share by block:")
    for g in sorted(masks):
        print(f"  {g:<10} {share[g] / norm2:6.1%}   ({int(masks[g].sum()):,} params)")

    demos = DemoSet.load(args.demos, hide_values=True)
    indices = np.arange(min(args.n, len(demos)))
    observations = demos.observations(indices)
    values = np.asarray(demos.values)[indices]
    distances = np.asarray(demos.distances)[indices].astype(int)
    feature_ids = np.asarray(demos.feature_ids)[indices].astype(int)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(indices))
    args.out.mkdir(parents=True, exist_ok=True)

    for group in sorted(masks):
        for offset in args.offsets:
            tag = f"{group}.{args.sweep}{offset_tag(offset)}"
            path = args.out / f"{tag}.npz"
            if path.exists():
                print(f"have {tag}")
                continue
            flat = base.flat + offset * np.where(masks[group], axis, 0.0)
            params = base.unravel(np.asarray(flat, dtype=np.float32))
            start = time.perf_counter()
            decoded = greedy_decode_cached(base.model, params, observations)
            outcomes = replay_all(demos, indices, decoded)
            reached = np.array([bool(o.get("reached_objective")) for o in outcomes])
            reached_fid = np.array(
                [-1 if o.get("reached_feature_id") is None else int(o["reached_feature_id"]) for o in outcomes]
            )
            np.savez(
                path,
                d_rich=distances[rows, richer],
                d_poor=distances[rows, 1 - richer],
                colour_of_rich=feature_ids[rows, richer],
                reached=reached,
                reached_fid=reached_fid,
            )
            print(f"{tag}: {time.perf_counter() - start:.0f}s  reached {reached.mean():.3f}")

    print("MASKED_DONE")


if __name__ == "__main__":
    main()
