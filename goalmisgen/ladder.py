"""Rungs of the base-checkpoint ladder — the same agent, earlier in its training.

The value axis is not a property of a checkpoint. It is a fitted slope,
``diff = drift + offset * axis``, and it exists only once a grid of arms has been
fine-tuned from a base. So "when does the axis appear?" cannot be answered by
loading checkpoints: it needs the whole sweep run again from each point in
training whose axis is wanted. Each of those points is a *rung*.

A rung is made an ordinary agent rather than a flag on the sweep. Everything
downstream resolves an agent as ``runs/<agent>/BASE.json`` — which checkpoint to
start from and what its objectives pay — so a rung that is an agent in that sense
needs no special case in the sweep driver, in ``014``, or in the manifest. It
costs one symlink back to the checkpoints it shares with the run it came from,
which is also why a rung takes no meaningful space.

The trap that symlink creates is worth stating, because it is silent. A rung's
``local-files`` holds *every* checkpoint of the parent run, so any tool that
resolves "the newest checkpoint" instead of reading ``BASE.json`` will fit the
final agent and label it with the rung's step count. Nothing in the analysis does
this today — ``014`` takes ``--base`` outright — and nothing should start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from goalmisgen.volume import parse_checkpoint_dirname, rung_agent_name


@dataclass(frozen=True)
class Rung:
    """One point in a run's training, prepared for a sweep."""

    agent: str
    """The rung's own agent name, e.g. ``novalue11.s1234.at70103040``."""

    source: str
    """The run it is a view of, e.g. ``novalue11.s1234``."""

    checkpoint: str
    """The checkpoint directory it starts from, e.g. ``cp_070103040``."""

    steps: int
    """How far into training that checkpoint is."""

    @property
    def checkpoint_path(self) -> Path:
        """Where the checkpoint sits inside the rung's own directory."""
        return Path("local-files") / self.checkpoint

    @property
    def label(self) -> str:
        """``70.1M`` — how a rung is named on an axis of a plot."""
        return f"{self.steps / 1_000_000:.1f}M"


def plan_rung(source: str, checkpoint: str) -> Rung:
    """Name a rung without touching the disk."""
    steps = parse_checkpoint_dirname(checkpoint)
    if steps is None:
        raise ValueError(f"{checkpoint!r} is not a checkpoint directory name, which looks like 'cp_070103040'")
    return Rung(agent=rung_agent_name(source, checkpoint), source=source, checkpoint=checkpoint, steps=steps)


def base_payload(rung: Rung, config: dict) -> dict:
    """The ``BASE.json`` a rung gets, from the checkpoint's own ``cfg.json``.

    ``steps`` is the *rung's* position in training, not the parent run's
    ``total_timesteps``. The shell version recorded the latter, so every rung of
    a 150M run claimed 150M steps — which is exactly the number a reader of a
    ladder wants and exactly the one it must not be. What the parent was aiming
    at is kept alongside, under a name that says so.
    """
    inner = config.get("cfg", config)
    return {
        "checkpoint": f"local-files/{rung.checkpoint}",
        "values": list(inner["train_env"]["objective_values"]),
        "objectives": inner["train_env"]["n_objectives"],
        "steps": rung.steps,
        "checkpoints_saved": 1,
        "rung_of": rung.source,
        "source_total_timesteps": inner.get("total_timesteps"),
    }


def rung_values(data: Path, source: str, checkpoint: str) -> tuple[float, ...]:
    """What the objectives pay at this rung, from the checkpoint's own cfg.json.

    Read rather than assumed so that a preview of a ladder describes the arms
    that would actually be trained, without having to write anything first.
    """
    config = json.loads((data / "runs" / source / "local-files" / checkpoint / "cfg.json").read_text())
    inner = config.get("cfg", config)
    return tuple(inner["train_env"]["objective_values"])


def make_rung(data: Path, source: str, checkpoint: str, *, dry_run: bool = False) -> Rung:
    """Prepare ``runs/<source>.at<steps>`` so the sweep driver can resolve it.

    Idempotent: a rung that already has its ``BASE.json`` is left exactly as it
    is, so a ladder interrupted half way can be re-run without disturbing the
    arms already fitted beneath it — the same idiom as the sweep driver and
    ``campaign.sh``.
    """
    rung = plan_rung(source, checkpoint)
    directory = data / "runs" / rung.agent
    marker = directory / "BASE.json"
    if marker.is_file():
        return rung

    checkpoints = data / "runs" / source / "local-files"
    if not (checkpoints / checkpoint).is_dir():
        available = sorted(p.name for p in checkpoints.glob("cp_*")) if checkpoints.is_dir() else []
        raise FileNotFoundError(
            f"{source} has no {checkpoint}. Checkpoint names are padded to the run's own width, "
            f"so a name copied from another run will not match here. Available: {', '.join(available) or 'none'}"
        )
    config = json.loads((checkpoints / checkpoint / "cfg.json").read_text())
    if dry_run:
        return rung

    directory.mkdir(parents=True, exist_ok=True)
    link = directory / "local-files"
    if not link.exists():
        # Relative, so the ladder survives the volume being mounted elsewhere.
        link.symlink_to(Path("..") / source / "local-files")
    marker.write_text(json.dumps(base_payload(rung, config), indent=2) + "\n")
    return rung


def discover_rungs(data: Path, agent: str) -> list[Rung]:
    """Every rung of one agent's ladder, deepest in training last.

    The agent itself is a rung -- the one every other is compared against -- so
    it is included, with its position read from the checkpoint its ``BASE.json``
    names rather than from that file's ``steps``. For a base agent those differ:
    ``steps`` is what the run was aiming at (150M) and the checkpoint is where it
    actually saved (140.2M), and a ladder plotted against the former would put
    its reference point 10M steps to the right of the weights it describes.
    """
    found: list[Rung] = []
    for directory in sorted((data / "runs").iterdir()) if (data / "runs").is_dir() else []:
        if directory.name != agent and not directory.name.startswith(f"{agent}.at"):
            continue
        marker = directory / "BASE.json"
        if not marker.is_file():
            continue
        checkpoint = Path(json.loads(marker.read_text())["checkpoint"]).name
        steps = parse_checkpoint_dirname(checkpoint)
        if steps is None:
            continue
        found.append(Rung(agent=directory.name, source=agent, checkpoint=checkpoint, steps=steps))
    return sorted(found, key=lambda rung: rung.steps)
