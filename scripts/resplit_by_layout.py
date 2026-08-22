"""Re-split an existing level dataset so that train and test share no maze.

    uv run python scripts/resplit_by_layout.py /workspace/data/levels/values/1.00-0.50@1M --dry-run
    uv run python scripts/resplit_by_layout.py /workspace/data/levels/values/*  --write

Rewrites ``split_train.npy`` / ``split_valid.npy`` / ``split_test.npy`` in place
and nothing else. Levels are untouched, so this costs a minute rather than a
regeneration, and the dataset's fingerprint is unaffected -- splits are stored
beside the level data and are not part of it.

Reports the layout leakage of the split it replaces, which is the number that
says whether the dataset needed this. An index-wise split of a pool whose
layouts recur seventeen times leaks essentially everything; a pool small enough
that every layout is unique leaks nothing and is left alone.

**Re-splitting a dataset invalidates results measured against its old split.**
Anything that reported held-out numbers from this pool was reading a test set
that is now a different set of levels. That is the point, but it means the
affected measurements have to be re-run rather than compared across the change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.envs.splits import layout_groups, layout_leakage, split_by_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", type=Path, nargs="+", help="Level dataset directories.")
    parser.add_argument("--seed", type=int, default=0, help="Chooses which layouts are held out.")
    parser.add_argument("--write", action="store_true", help="Actually rewrite the split files.")
    parser.add_argument("--dry-run", action="store_true", help="Report and change nothing (the default).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write and args.dry_run:
        raise SystemExit("--write and --dry-run ask for opposite things")

    for directory in args.datasets:
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            print(f"{directory}: not a level dataset, skipping")
            continue
        meta = json.loads(meta_path.read_text())
        names = meta.get("splits") or []
        if not names:
            print(f"{directory.name}: has no stored splits, skipping")
            continue

        walls = np.load(directory / "walls_packed.npy", mmap_mode="r")
        groups = layout_groups(walls)
        n, layouts = len(groups), int(groups.max()) + 1
        old = {name: np.load(directory / f"split_{name}.npy") for name in names}
        before = layout_leakage(old, groups) if "train" in old else {}

        print(f"\n{directory.name}: {n:,} levels, {layouts:,} distinct layouts ({n / layouts:.1f} levels each)")
        for name, value in sorted(before.items()):
            print(f"  before: {value:6.1%} of {name} shares a layout with train  ({len(old[name]):,} levels)")

        new = split_by_layout(
            walls, valid=len(old.get("valid", [])) or 50_000, test=len(old.get("test", [])) or 50_000, seed=args.seed
        )
        after = layout_leakage(new, groups)
        for name in sorted(after):
            held = len(np.unique(groups[new[name]]))
            print(
                f"  after : {after[name]:6.1%} of {name} shares a layout with train  ({len(new[name]):,} levels, {held:,} layouts)"
            )
        print(f"  train : {len(new['train']):,} levels, {len(np.unique(groups[new['train']])):,} layouts")

        if not args.write:
            print("  (dry run; pass --write to rewrite the split files)")
            continue
        for name, indices in new.items():
            np.save(directory / f"split_{name}.npy", indices)
        meta["splits"] = sorted(new)
        meta["split_by"] = "layout"
        meta["split_seed"] = args.seed
        meta_path.write_text(json.dumps(meta, indent=2))
        print("  rewritten")


if __name__ == "__main__":
    main()
