"""Index every dataset and every trained agent on the data volume.

    uv run python scripts/manifest.py --data /workspace/data --notes RUNS.toml --out MANIFEST.md

Walks the volume, reads each dataset's fingerprint and each run's saved
configuration, and writes a table linking them. The point is that a run's
configuration is already recoverable — ``013`` writes the whole thing into the
run directory — while the *reason it was run* is not, and that is the part that
decays. A directory named ``threeobj2`` says nothing about why it exists
alongside ``threeobj``.

So a run's note comes from ``RUNS.toml`` in the repository, falling back to a
``NOTE.md`` written beside its checkpoints at launch. The repository is the
source of truth because the volume is rented and the repository is not. Runs
matching neither are listed as unexplained rather than quietly omitted: an
unaccounted 2 GB of checkpoints is exactly what this is meant to surface.

The manifest is generated rather than maintained by hand, and committed, so it
can be regenerated after any run and the diff shows what changed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--notes", type=Path, default=Path("RUNS.toml"), help="Why each run and dataset exists.")
    parser.add_argument("--out", type=Path, default=Path("MANIFEST.md"))
    parser.add_argument("--generated-at", type=str, default=None, help="Timestamp to record; defaults to now, UTC.")
    return parser.parse_args()


def human(size: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.0f}T"


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def load_notes(path: Path) -> list[tuple[str, str]]:
    """Patterns and their notes, in file order so the most specific can come first."""
    if not path.is_file():
        return []
    entries = tomllib.loads(path.read_text()).get("entry", [])
    return [(entry["match"], " ".join(entry["note"].split())) for entry in entries]


def note_for(relative: Path, notes: list[tuple[str, str]], run: Path | None = None) -> str:
    """The first matching note, else whatever was written beside the checkpoints."""
    for pattern, note in notes:
        if fnmatch.fnmatch(str(relative), pattern):
            return note
    if run is not None:
        for candidate in (run / "NOTE.md", run / "note.md"):
            if candidate.is_file():
                return " ".join(candidate.read_text().split()) + " _(from NOTE.md)_"
    return ""


def run_config(run: Path) -> dict:
    """A run's environment settings, from whichever config it saved.

    ``013`` writes the reset checkpoint's config under ``init``; a base training
    run has one inside each checkpoint. Either way the values, objective count
    and level dataset are recorded, so a run can always say what task it was on
    even when nobody wrote down why.
    """
    for candidate in sorted(run.glob("init/cfg.json")) + sorted(run.glob("local-files/cp_*/cfg.json"), reverse=True):
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = payload.get("cfg", payload)
        env = config.get("train_env", {})
        return {
            "objectives": env.get("n_objectives"),
            "values": env.get("objective_values"),
            "levels": (env.get("level_dataset") or "").rsplit("/", 1)[-1],
            "steps": config.get("total_timesteps"),
            "lr": config.get("learning_rate"),
            "hidden_values": env.get("colour_is_the_only_value_cue"),
        }
    return {}


def datasets(root: Path, notes: list[tuple[str, str]]) -> list[dict]:
    found = []
    for meta in sorted(root.rglob("meta.json")):
        try:
            payload = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        values = None
        values_file = meta.parent / "values.npy"
        if values_file.is_file():
            try:
                import numpy as np

                values = tuple(sorted({float(v) for v in np.load(values_file)[0]}, reverse=True))
            except Exception:  # a dataset that cannot be read is still worth listing
                values = None
        found.append(
            {
                "path": meta.parent.relative_to(root),
                "levels": payload.get("n_levels"),
                "fingerprint": payload.get("fingerprint"),
                "size": human(directory_size(meta.parent)),
                "values": values,
                "note": note_for(meta.parent.relative_to(root), notes),
            }
        )
    return found


def runs(root: Path, notes: list[tuple[str, str]]) -> list[dict]:
    found = []
    for local in sorted(root.rglob("local-files")):
        run = local.parent
        checkpoints = sorted(local.glob("cp_*"))
        found.append(
            {
                "path": run.relative_to(root),
                "checkpoints": len(checkpoints),
                "last": checkpoints[-1].name if checkpoints else "none",
                "size": human(directory_size(run)),
                "note": note_for(run.relative_to(root), notes, run),
                **run_config(run),
            }
        )
    return found


def main() -> None:
    args = parse_args()
    stamp = args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    notes = load_notes(args.notes)
    all_datasets, all_runs = datasets(args.data, notes), runs(args.data, notes)

    # A group note repeated down sixteen rows is noise; numbering them keeps the
    # table scannable and puts each reason in one place, where it can be edited
    # once.
    legend: dict[str, int] = {}

    def cite(note: str) -> str:
        if not note:
            return "**none**"
        return f"[{legend.setdefault(note, len(legend) + 1)}]"

    lines = [
        "# What is on the data volume",
        "",
        f"Generated by `scripts/manifest.py` at {stamp}. Regenerate after any run;",
        "the diff is the record of what changed.",
        "",
        "## Level datasets",
        "",
        "Values are read from the stored levels themselves rather than from a name, so a",
        "mislabelled directory shows up here. The fingerprint covers the sampler and the",
        "source of every module that decides what a level contains, so two datasets",
        "sharing one are interchangeable.",
        "",
        "| dataset | levels | objective values | fingerprint | size | why |",
        "|---|---|---|---|---|---|",
    ]
    for entry in all_datasets:
        values = ", ".join(f"{v:g}" for v in entry["values"]) if entry["values"] else "?"
        count = f"{entry['levels']:,}" if entry["levels"] else "?"
        lines.append(
            f"| `{entry['path']}` | {count} | {values} | `{entry['fingerprint']}` | {entry['size']} | {cite(entry['note'])} |"
        )

    lines += [
        "",
        "## Trained agents",
        "",
        "Every run records its own configuration, so the task is always recoverable. The",
        "note is not recoverable and is the column that matters: a run without one is a",
        "directory of checkpoints nobody can explain.",
        "",
        "| run | objectives | values | levels | steps | checkpoints | size | why it was run |",
        "|---|---|---|---|---|---|---|---|",
    ]
    unexplained = 0
    for entry in all_runs:
        values = ", ".join(f"{v:g}" for v in entry["values"]) if entry.get("values") else "?"
        steps = f"{entry['steps']:,}" if entry.get("steps") else "?"
        unexplained += 0 if entry["note"] else 1
        lines.append(
            f"| `{entry['path']}` | {entry.get('objectives') or '?'} | {values} | `{entry.get('levels') or '?'}` | "
            f"{steps} | {entry['checkpoints']} | {entry['size']} | {cite(entry['note'])} |"
        )

    lines += ["", "## Why", ""]
    for note, number in sorted(legend.items(), key=lambda item: item[1]):
        lines += [f"**[{number}]** {note}", ""]

    if unexplained:
        lines += [
            "",
            f"{unexplained} run(s) have no note. Add a `NOTE.md` beside their checkpoints, or",
            "pass `--note` when launching so it is written at the time rather than",
            "reconstructed later.",
        ]

    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}  ({len(all_datasets)} datasets, {len(all_runs)} runs, {unexplained} unexplained)")


if __name__ == "__main__":
    main()
