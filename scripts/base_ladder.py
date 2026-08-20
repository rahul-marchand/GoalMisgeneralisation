"""Fit the value axis from several points in one agent's own training.

    uv run python scripts/base_ladder.py --agent novalue11.s1234 --near 20 40
    uv run python scripts/base_ladder.py --agent novalue11.s1234 --near 2 5 10 15 20 30 40 --dry-run

``value_axis_sweep.py`` fits one axis, from whichever checkpoint the agent's
``BASE.json`` names. This runs it from a ladder of earlier checkpoints instead,
which is the only way to ask *when* the axis appears: the axis is a fitted slope
over a grid of fine-tunes, so it has to be re-fitted at every point in training
it is wanted at, and no amount of loading checkpoints substitutes for that.

Two things it does that the shell version in ``campaign.sh`` could not:

* **Checkpoints are resolved by step count, not by string.** ``--near 20`` asks
  for the saved checkpoint closest to 20M steps. cleanba pads checkpoint names to
  the run's own width and saves on an inherited schedule, so the name for "about
  20M" is ``cp_020029440`` in one run and ``cp_20029440`` in another, and neither
  is guessable. Asking for the number and reporting what was found removes a
  whole class of rung that silently does not run — which is what happened to the
  100M rung, requested as ``cp_100146560`` against a real ``cp_100147200``.
* **Rungs are ordinary agents.** See ``goalmisgen/ladder.py``.

Both stages skip whatever is already on disk, so a ladder can be split across
several GPUs by giving each one a different ``--near``, and re-run after an
interruption without repeating work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.design import estimated_hours, sweep_arms  # noqa: E402
from goalmisgen.ladder import Rung, make_rung, rung_values  # noqa: E402
from goalmisgen.volume import arm_is_complete, parse_checkpoint_dirname  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--agent", type=str, required=True, help="The run whose training the ladder climbs.")
    parser.add_argument(
        "--near",
        type=float,
        nargs="+",
        default=None,
        help="Rungs, in millions of steps. Each resolves to the nearest saved checkpoint, which is "
        "reported so a rung that landed somewhere unintended is visible rather than assumed.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Rungs named outright, e.g. cp_020029440. Use --near unless reproducing an exact ladder.",
    )
    parser.add_argument(
        "--objectives",
        type=int,
        nargs="+",
        default=None,
        help="Which objectives to sweep at each rung. Defaults to every objective the agent has. "
        "Sweeping both is what makes cos(axis_0, axis_1) readable per rung, i.e. whether the "
        "one-knob structure is there from the start or arrives.",
    )
    parser.add_argument("--steps", type=int, default=400_000, help="Per arm. Must match the rung it is compared with.")
    parser.add_argument("--offsets", type=float, nargs="+", default=None, help="Passed through to the sweep.")
    parser.add_argument("--arm-levels", type=int, default=150_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoints-per-arm", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the ladder, run nothing.")
    return parser.parse_args()


def saved_checkpoints(data: Path, agent: str) -> dict[int, str]:
    """Every checkpoint the run wrote, keyed by the step it was written at."""
    directory = data / "runs" / agent / "local-files"
    if not directory.is_dir():
        sys.exit(f"{directory} does not exist, so {agent} has no checkpoints to climb")
    found = {}
    for path in directory.glob("cp_*"):
        steps = parse_checkpoint_dirname(path.name)
        if steps is not None and path.is_dir():
            found[steps] = path.name
    if not found:
        sys.exit(f"{directory} holds no cp_* directories")
    return found


def resolve(data: Path, agent: str, near: list[float] | None, named: list[str] | None) -> list[tuple[str, float | None]]:
    """The checkpoint for each requested rung, nearest-first, de-duplicated.

    Two requested rungs can land on one checkpoint — a run saves every ~1M steps
    early and every ~10M later, so ``--near 22 25`` is one rung, not two. Saying
    so is the point: silently sweeping the same base twice would put two
    identical axes on the plot and read as a replication.
    """
    available = saved_checkpoints(data, agent)
    requested: list[tuple[str, float | None]] = []
    for name in named or []:
        if parse_checkpoint_dirname(name) is None:
            sys.exit(f"{name!r} is not a checkpoint directory name, which looks like 'cp_070103040'")
        requested.append((name, None))
    for millions in near or []:
        target = millions * 1_000_000
        closest = min(available, key=lambda steps: abs(steps - target))
        requested.append((available[closest], millions))

    seen: dict[str, float | None] = {}
    for name, asked in requested:
        if name in seen:
            print(f"  note: {asked}M and {seen[name]}M both resolve to {name}; keeping one rung")
            continue
        seen[name] = asked
    return list(seen.items())


def preview(args: argparse.Namespace, rung: Rung) -> tuple[int, int]:
    """What one rung would cost, without writing anything. (to train, already done)"""
    values = rung_values(args.data, rung.source, rung.checkpoint)
    objectives = args.objectives if args.objectives is not None else list(range(len(values)))
    offsets = tuple(args.offsets) if args.offsets else None
    arms = [arm for objective in objectives for arm in sweep_arms(objective, offsets=offsets)]
    done = sum(
        arm_is_complete(args.data / "runs" / rung.agent / "arms" / arm.dirname(args.steps), args.steps) for arm in arms
    )
    return len(arms) - done, done


def sweep(args: argparse.Namespace, rung: Rung) -> bool:
    """Run one rung's sweep. Returns whether it succeeded."""
    command = [
        "uv", "run", "python", "scripts/value_axis_sweep.py",
        "--data", str(args.data),
        "--agent", rung.agent,
        "--steps", str(args.steps),
        "--arm-levels", str(args.arm_levels),
        "--lr", str(args.lr),
        "--checkpoints", str(args.checkpoints_per_arm),
    ]  # fmt: skip
    if args.objectives is not None:
        command += ["--objectives", *[str(o) for o in args.objectives]]
    if args.offsets is not None:
        command += ["--offsets", *[f"{o:g}" for o in args.offsets]]
    if args.dry_run:
        command.append("--dry-run")
    print(f"\n=== rung {rung.label} ({rung.checkpoint}) -> {rung.agent} ===", flush=True)
    return subprocess.run(command, cwd=Path(__file__).resolve().parent.parent).returncode == 0


def main() -> None:
    args = parse_args()
    if not args.near and not args.checkpoints:
        sys.exit("give --near (millions of steps) or --checkpoints (exact names); there is no default ladder")

    rungs = []
    for name, asked in resolve(args.data, args.agent, args.near, args.checkpoints):
        rung = make_rung(args.data, args.agent, name, dry_run=args.dry_run)
        drift = "" if asked is None else f"  (asked {asked}M, off by {abs(rung.steps - asked * 1e6) / 1e6:.2f}M)"
        print(f"  {rung.label:>7}  {rung.checkpoint}  ->  {rung.agent}{drift}")
        rungs.append(rung)

    rungs.sort(key=lambda r: r.steps)
    print(f"\n{len(rungs)} rung{'' if len(rungs) == 1 else 's'}: {', '.join(r.label for r in rungs)}")

    if args.dry_run:
        # Previewed here rather than delegated, because the sweep driver resolves
        # an agent through the BASE.json a dry run has deliberately not written.
        print(f"\n{'rung':>8}{'to train':>10}{'done':>8}{'hours':>8}   agent")
        total = 0
        for rung in rungs:
            todo, done = preview(args, rung)
            total += todo
            print(f"{rung.label:>8}{todo:>10}{done:>8}{estimated_hours(todo, args.steps):>8.2f}   {rung.agent}")
        print(f"\n{total} arms to train, ~{estimated_hours(total, args.steps):.1f} h on one 4090")
        print("Nothing was written. Drop --dry-run to run it.")
        return

    failed = [rung.label for rung in rungs if not sweep(args, rung)]
    if failed:
        sys.exit(f"\nLADDER INCOMPLETE -- these rungs failed: {', '.join(failed)}")
    print(f"\nLADDER COMPLETE -- {len(rungs)} rungs at {args.steps:,} steps per arm")


if __name__ == "__main__":
    main()
