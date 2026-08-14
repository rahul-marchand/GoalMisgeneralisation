"""Fine-tune one agent onto the wide value grid in ``goalmisgen/design.py``.

    uv run python scripts/value_axis_sweep.py --agent novalue11.s1234 --objectives 0 1
    uv run python scripts/value_axis_sweep.py --agent novalue11.s1234 --dry-run

Replaces ``scripts/value_axis_grid.sh``, which hardcoded seven values, one
agent's checkpoint path and one campaign's directory layout. What changes here
is not the mechanics — it is still one ``013`` per arm — but three things the
shell version could not do:

* **The grid comes from the design rather than from the script.** Offsets, their
  count and their spacing are the experiment; see the leverage argument in
  ``goalmisgen/design.py``. Changing them there changes every sweep, and the
  tests there assert the properties the argument depends on.
* **The base checkpoint comes from ``BASE.json``**, not from a default path with
  a step number baked into it. ``threeobj.even``'s canonical checkpoint is
  ``cp_70103040`` rather than the 80M it trained for, and a rule that reads the
  file gets that right where a hardcoded string got it right only by accident.
* **Level datasets are shared.** They are keyed by what the objectives pay and
  how many levels there are, so a value already generated for another seed or
  another campaign is reused rather than regenerated. ``FixedValues`` consumes no
  randomness, so datasets differing only in value have byte-identical layouts.

Both stages skip whatever is already on disk, so an interrupted sweep can be
re-run without repeating work or half-writing a dataset. Arms run one at a time:
there is one GPU, and arms that shared it would not be the same fine-tune.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.design import arm_values, check_no_preference_flip, leverage, sweep_arms  # noqa: E402
from goalmisgen.volume import dataset_dirname  # noqa: E402

# Median ``charts/0/SPS`` measured on an RTX 4090, which is what the campaign
# runs on. The original 150M runs managed 4,200 on whatever they used; the
# bottleneck is the learner update, so a faster card converts almost linearly
# into throughput and a 4090 is 2.33x the original. Used only to print an
# estimate before a sweep commits hours to it.
MEASURED_SPS = 9_800

# Seconds to construct one level, by objective count: a maze plus a
# breadth-first search per objective, and a mutual reachability check beyond two.
SECONDS_PER_LEVEL = {2: 0.00042, 3: 0.00175}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--agent", type=str, required=True, help="A directory under runs/, e.g. novalue11.s1234.")
    parser.add_argument(
        "--objectives",
        type=int,
        nargs="+",
        default=None,
        help="Which objectives to sweep. Defaults to every objective the agent has. Sweeping "
        "more than one is what separates a value from the gap between two values: nothing "
        "behavioural can, since the choice turns on the difference alone.",
    )
    parser.add_argument("--steps", type=int, default=750_000, help="Per arm. Shorter arms have been the cleaner ones.")
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=None,
        help="The positive side of the grid; each is mirrored. Defaults to the design in "
        "goalmisgen/design.py. Give this to reproduce an earlier grid exactly, which a "
        "replication has to do.",
    )
    parser.add_argument(
        "--allow-reorder",
        action="store_true",
        help="Permit arms whose values reorder the objectives. Refused by default: with two "
        "objectives that is the agent being asked to hold the opposite preference, and the "
        "writes fail there. With three it is deliberate -- the grid has to span rank changes, "
        "because a task whose choice reduces to one difference cannot need two dimensions.",
    )
    parser.add_argument("--arm-levels", type=int, default=150_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the estimate, run nothing.")
    return parser.parse_args()


def is_complete(run_dir: Path, steps: int) -> bool:
    """Did this arm reach the length its name claims?

    Presence of *a* checkpoint is not enough. An arm killed part way leaves the
    ones it had already saved, and resuming would skip it and then fit it as
    though it had run the full budget -- a 200k arm sitting in the grid under a
    @400k name, which is precisely the silent incomparability the naming scheme
    exists to prevent. Checkpoints are named in steps, so the last one says how
    far the arm actually got.
    """
    saved = [int(p.name[3:]) for p in run_dir.glob("local-files/cp_*") if p.name[3:].isdigit()]
    return bool(saved) and max(saved) >= 0.98 * steps


def read_base(data: Path, agent: str) -> tuple[Path, tuple[float, ...]]:
    """The checkpoint every arm starts from, and what its objectives pay."""
    directory = data / "runs" / agent
    marker = directory / "BASE.json"
    if not marker.is_file():
        sys.exit(
            f"{marker} does not exist. It is written by scripts/migrate_volume.py --retire, "
            "and records which checkpoint of this agent everything downstream resolves."
        )
    payload = json.loads(marker.read_text())
    checkpoint = directory / payload["checkpoint"]
    if not checkpoint.is_dir():
        sys.exit(f"{marker} points at {payload['checkpoint']}, which is not there")
    return checkpoint, tuple(payload["values"])


def ensure_dataset(data: Path, values: tuple[float, ...], n_levels: int, size: int, dry_run: bool) -> Path:
    """One shared dataset per value tuple, generated only if nobody has already."""
    directory = data / "levels" / "values" / dataset_dirname(values, n_levels)
    if directory.is_dir():
        print(f"  present  {directory.name}")
        return directory
    print(f"  generate {directory.name}")
    if dry_run:
        return directory
    # 5k holdouts rather than the base datasets' 50k: an arm this short sees each
    # training level about once, so spending a third of the dataset on holdouts
    # it never reads would only shrink what it trains on.
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/generate_levels.py",
            "--n-levels",
            str(n_levels),
            "--min-size",
            str(size),
            "--max-size",
            str(size),
            "--valid-levels",
            "5000",
            "--test-levels",
            "5000",
            "--n-objectives",
            str(len(values)),
            "--objective-values",
            *[f"{v:g}" for v in values],
            "--out",
            str(directory),
        ],
        check=True,
    )
    return directory


def main() -> None:
    args = parse_args()
    checkpoint, base_values = read_base(args.data, args.agent)
    objectives = args.objectives if args.objectives is not None else list(range(len(base_values)))

    offsets = tuple(args.offsets) if args.offsets else None
    arms = [arm for objective in objectives for arm in sweep_arms(objective, offsets=offsets)]
    problems = check_no_preference_flip(base_values, arms)
    if problems and args.allow_reorder:
        print(f"  {len(problems)} arms reorder the objectives, allowed explicitly")
        problems = []
    if problems:
        print("These arms reorder the objectives, which makes them a different task:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit("Narrow the offsets in goalmisgen/design.py, or sweep a different objective.")

    hours = len(arms) * args.steps / MEASURED_SPS / 3600
    print(f"agent        {args.agent}")
    print(f"checkpoint   {checkpoint.name}")
    print(f"base values  {base_values}")
    print(f"objectives   {objectives}")
    for objective in objectives:
        offsets = [a.offset for a in arms if a.objective == objective]
        print(f"  o{objective}: {len(offsets)} arms, leverage {leverage(offsets):.3f}")
    print(f"arms         {len(arms)} x {args.steps:,} steps  (~{hours:.1f} h at {MEASURED_SPS:,} SPS)")

    print("\n=== levels ===")
    datasets = {}
    for arm in arms:
        values = arm_values(base_values, arm)
        if values not in datasets:
            datasets[values] = ensure_dataset(args.data, values, args.arm_levels, args.size, args.dry_run)

    # Generation is CPU work in front of the GPU, and it is paid once: the
    # library is keyed by what the objectives pay, so the second agent swept at
    # this design reuses every dataset the first one generated.
    missing = [d for d in datasets.values() if not d.is_dir()]
    if missing:
        seconds = len(missing) * args.arm_levels * SECONDS_PER_LEVEL[min(len(base_values), 3)]
        print(f"\n  {len(missing)} of {len(datasets)} need generating, ~{seconds / 60:.0f} min, once for all agents")

    print(f"\n=== arms ({len(arms)}) ===")
    for index, arm in enumerate(arms, start=1):
        values = arm_values(base_values, arm)
        run_dir = args.data / "runs" / args.agent / "arms" / arm.dirname(args.steps)
        if is_complete(run_dir, args.steps):
            print(f"  [{index:>3}/{len(arms)}] {run_dir.name} already complete")
            continue
        if run_dir.exists():
            reached = [int(p.name[3:]) for p in run_dir.glob("local-files/cp_*") if p.name[3:].isdigit()]
            print(
                f"  [{index:>3}/{len(arms)}] {run_dir.name} stopped at "
                f"{max(reached, default=0):,} of {args.steps:,}, retraining"
            )
            if not args.dry_run:
                shutil.rmtree(run_dir)
        print(f"  [{index:>3}/{len(arms)}] {run_dir.name}  values {values}  seed {arm.seed}")
        if args.dry_run:
            continue
        logs = args.data / "runs" / args.agent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / f"{arm.dirname(args.steps)}.log").open("w") as log:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "experiments/013_value_axis.py",
                    str(checkpoint),
                    "--objective-values",
                    *[f"{v:g}" for v in values],
                    "--levels",
                    str(datasets[values]),
                    "--run-dir",
                    str(run_dir),
                    "--steps",
                    str(args.steps),
                    "--lr",
                    str(args.lr),
                    "--checkpoints",
                    str(args.checkpoints),
                    "--size",
                    str(args.size),
                    "--seed",
                    str(arm.seed),
                    "--note",
                    f"{arm.sweep}{arm.offset:+.2f} of the wide value grid on {args.agent}.",
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    print("\nSWEEP_COMPLETE" if not args.dry_run else "\nThis was a plan. Drop --dry-run to run it.")


if __name__ == "__main__":
    main()
