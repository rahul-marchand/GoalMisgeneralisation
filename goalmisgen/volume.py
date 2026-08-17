"""What things on the data volume are called, and how to read a name back.

A directory name is the only piece of metadata that is visible before anything
is loaded, so it is the only one that can stop a mistake rather than record it.
The previous layout recorded two facts in prose that it should have carried in
the path, and both of them cost something:

* **Which agent an arm belongs to.** ``valueaxis/runs/v070`` was seed 1234 and
  ``valueaxis_s5678/runs/v070`` was seed 5678, so one seed was named and the
  other was a convention. A third seed has nowhere to go that is not a third
  convention.
* **How long an arm trained for.** Seed 1234's arms ran 3M steps and seed
  5678's ran 750k, and every quantity that scales with fine-tune length is
  therefore incomparable between them. ``results/seed-comparison.txt`` says so
  in a paragraph, which works exactly until somebody does not read it.

So an arm is ``<sweep><offset>@<steps>`` beneath the agent it was fine-tuned
from, and the two things that must never be silently mixed are both in the
name. ``014`` already refuses to fit arms at different budgets in one directory
(``arm_checkpoints``); putting the budget in the path extends that refusal
across directories, where the comparison is not checked at all.

The scheme is otherwise deliberately flat. Arms from a seven-point grid and a
thirty-point grid at the same length belong in the same directory and *should*
be fitted together — more leverage on the same axis — so design generation is
not part of the name. Only length is, because only length makes arms
incompatible.

One wart worth knowing: ``+`` sorts before ``-`` in ASCII, so a directory
listing puts every negative offset last. Nothing depends on it — ``014`` keys
arms by their parsed value — but anything presenting arms in offset order has to
parse the name rather than sort the listing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ``o1+020@750k``: the colour-1 sweep, trained at 0.20 above the base value, for
# 750,000 steps. Composition arms that move several values at once do not have a
# single offset and are named separately; see ``is_composition_arm``.
_ARM = re.compile(r"^(?P<sweep>[a-z][a-z0-9]*)(?P<sign>[+-])(?P<offset>\d{3})@(?P<steps>\d+[kM])$")


@dataclass(frozen=True)
class ArmName:
    """One fine-tuning arm, as its directory name says it is."""

    sweep: str
    offset: float
    steps: int

    @property
    def is_null(self) -> bool:
        """Fine-tuned onto the value it already had, so it measures drift."""
        return self.offset == 0.0


def _round_tag(n: int, what: str) -> str:
    """``750000`` -> ``750k``, ``3000000`` -> ``3M``.

    Only exact multiples get the short form, so a tag can always be read back to
    the number that produced it. An arm at 1,234,567 steps would round to the
    same tag as one at 1.2M and silently claim to be comparable with it, which
    is the failure this whole module exists to prevent.
    """
    if n <= 0:
        raise ValueError(f"{what} cannot be {n}")
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    raise ValueError(
        f"{n:,} has no exact tag, so it cannot name {what}; these should be round numbers "
        "so that two things claiming the same size really have it"
    )


def _parse_round_tag(tag: str, what: str) -> int:
    match = re.fullmatch(r"(\d+)([kM])", tag)
    if match is None:
        raise ValueError(f"{tag!r} is not {what}, which looks like '750k' or '3M'")
    return int(match.group(1)) * (1_000 if match.group(2) == "k" else 1_000_000)


def steps_tag(steps: int) -> str:
    """How long an arm trained for, as it appears in the arm's directory name."""
    return _round_tag(steps, "a number of training steps")


def parse_steps_tag(tag: str) -> int:
    """``750k`` -> ``750000``. The inverse of :func:`steps_tag`."""
    return _parse_round_tag(tag, "a step tag")


def offset_tag(offset: float) -> str:
    """``+0.2`` -> ``+020``, ``-0.05`` -> ``-005``, ``0.0`` -> ``+000``.

    Signed always, so the null arm sorts with the rest of its sweep rather than
    reading as a different kind of thing, and so a missing sign is a malformed
    name rather than an assumed positive.
    """
    hundredths = round(offset * 100)
    if abs(hundredths / 100 - offset) > 1e-9:
        raise ValueError(f"offset {offset} is finer than the hundredth of a value the name can carry")
    return f"{'-' if hundredths < 0 else '+'}{abs(hundredths):03d}"


def arm_dirname(sweep: str, offset: float, steps: int) -> str:
    """The directory one arm lives in, e.g. ``o1+020@750k``."""
    if not re.fullmatch(r"[a-z][a-z0-9]*", sweep):
        raise ValueError(f"sweep {sweep!r} should be a short lower-case tag like 'o0', 'o1' or 'o2'")
    return f"{sweep}{offset_tag(offset)}@{steps_tag(steps)}"


def parse_arm_dirname(name: str) -> ArmName | None:
    """Read an arm's name back, or ``None`` if it is not one.

    Returning ``None`` rather than raising is deliberate: callers walk a
    directory that also holds composition arms and whatever else has been put
    there, and the useful behaviour is to report what was skipped rather than to
    stop on the first thing that is not an arm.
    """
    match = _ARM.fullmatch(name)
    if match is None:
        return None
    offset = int(match.group("offset")) / 100
    return ArmName(
        sweep=match.group("sweep"),
        offset=-offset if match.group("sign") == "-" else offset,
        steps=parse_steps_tag(match.group("steps")),
    )


def is_composition_arm(name: str) -> bool:
    """``m_120_045_030@1M`` — several values moved at once.

    These are held out of every fit by construction, so they are named for the
    values they carry rather than for an offset they do not have.
    """
    return re.fullmatch(r"m(_\d{3})+@\d+[kM]", name) is not None


def composition_arm_dirname(values: Iterable[float], steps: int) -> str:
    """``m_120_045_030@1M`` from the values it was trained on."""
    tags = "".join(f"_{round(v * 100):03d}" for v in values)
    if not tags:
        raise ValueError("a composition arm needs at least one value")
    return f"m{tags}@{steps_tag(steps)}"


def composition_arm_values(name: str) -> tuple[float, ...] | None:
    """The values a composition arm was trained at, or ``None`` if it is not one.

    The inverse of :func:`composition_arm_dirname`. Composition arms carry their
    values outright rather than an offset, because they move several at once and
    so have no single offset to be named for.
    """
    if not is_composition_arm(name):
        return None
    return tuple(int(part) / 100 for part in name.split("@")[0].split("_")[1:])


def arm_trained_values(name: str, base_values: tuple[float, ...]) -> tuple[float, ...] | None:
    """What an arm's objectives paid, from its directory name alone.

    Reading them from the name keeps the analysis honest if a grid is edited: a
    table in a script can drift out of step with what was actually run, a
    directory name cannot.
    """
    parsed = parse_arm_dirname(name)
    if parsed is not None:
        index = int(parsed.sweep[1:])
        if not 0 <= index < len(base_values):
            return None
        values = list(base_values)
        values[index] = round(values[index] + parsed.offset, 10)
        return tuple(values)
    return composition_arm_values(name)


def values_tag(values: Iterable[float]) -> str:
    """``(1.0, 0.5)`` -> ``1.00-0.50``, the key of a shared level dataset.

    Datasets are keyed by what the objectives pay because that is the only thing
    that distinguishes them: ``FixedValues`` consumes no randomness, so two
    datasets generated at one seed with different values have byte-identical
    layouts. Keying by content rather than by the campaign that first needed it
    is what lets one dataset serve every seed.
    """
    values = list(values)
    if not values:
        raise ValueError("a level dataset needs at least one objective value")
    if any(v < 0 for v in values):
        raise ValueError(f"objective values are rewards and cannot be negative: {values}")
    return "-".join(f"{v:.2f}" for v in values)


def parse_values_tag(tag: str) -> tuple[float, ...]:
    """``1.00-0.50`` -> ``(1.0, 0.5)``. The inverse of :func:`values_tag`."""
    if not re.fullmatch(r"\d+\.\d{2}(-\d+\.\d{2})*", tag):
        raise ValueError(f"{tag!r} is not a values tag like '1.00-0.50'")
    return tuple(float(part) for part in tag.split("-"))


def dataset_dirname(values: Iterable[float], n_levels: int) -> str:
    """``1.00-0.50@1M`` — the key one shared level dataset lives under.

    The level count is part of the key and not decoration. Two datasets at the
    same values but different counts are *not* interchangeable: the layouts are
    a deterministic function of the seed and the number of levels asked for, so
    a 500k dataset is not the first half of a 1M one. Keying on values alone
    would quietly merge ``levels11`` with ``valueaxis/levels/v050`` — same
    values, 1M levels against 500k — and every run downstream would train on
    mazes it did not train on.

    What *is* shared is everything at one key: ``FixedValues`` consumes no
    randomness, so datasets differing only in what the objectives pay have
    byte-identical layouts, and one copy serves every seed and every campaign.
    """
    return f"{values_tag(values)}@{_round_tag(n_levels, 'a number of levels')}"


def parse_dataset_dirname(name: str) -> tuple[tuple[float, ...], int]:
    """``1.00-0.50@1M`` -> ``((1.0, 0.5), 1000000)``."""
    tag, _, count = name.partition("@")
    if not count:
        raise ValueError(f"{name!r} is missing its level count, which is part of what makes a dataset shareable")
    return parse_values_tag(tag), _parse_round_tag(count, "a level count")


def discover_arms(
    arms: Path,
    objective: int,
    base_value: float,
    steps: int | None = None,
    at: int = -1,
) -> dict[float, Path]:
    """One sweep's arms, keyed by the value each was trained at.

    The key is the value rather than the offset because that is what the analysis
    has always regressed against, so the scripts that consume this did not have
    to change shape when the directories were renamed.

    ``steps`` is not optional in spirit. An agent's ``arms/`` directory now holds
    every sweep ever run against it, and arms of different lengths are not
    comparable -- that is the whole reason the length is in the name. So a
    directory holding more than one length raises unless the caller says which it
    means. ``014`` already warned about this within a directory; this makes the
    mistake unavailable rather than reported.
    """
    if not arms.is_dir():
        return {}
    found: dict[float, Path] = {}
    lengths: set[int] = set()
    for run in sorted(arms.iterdir()):
        parsed = parse_arm_dirname(run.name)
        if parsed is None or parsed.sweep != f"o{objective}":
            continue
        if steps is not None and parsed.steps != steps:
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            continue
        try:
            found[round(base_value + parsed.offset, 10)] = checkpoints[at]
        except IndexError:
            print(f"  {run.name}: no checkpoint at index {at}, skipping")
            continue
        lengths.add(parsed.steps)
    if steps is None and len(lengths) > 1:
        raise ValueError(
            f"{arms} holds arms at {sorted(lengths)} steps. Fitting them together would mix "
            "budgets, which is what putting the length in the name exists to prevent -- pass "
            "steps= to say which sweep you mean."
        )
    return found


def arm_lengths(arms: Path) -> set[int]:
    """Every arm budget present in one agent's ``arms/`` directory."""
    if not arms.is_dir():
        return set()
    return {p.steps for run in arms.iterdir() if (p := parse_arm_dirname(run.name)) is not None}


def sweep_index(prefix: str) -> int:
    """Which objective a sweep moves, from however it is named on the command line.

    ``v`` and ``c`` are the old two-objective names -- ``v`` swept colour 1 and
    ``c`` swept colour 0 -- and are still accepted so that recorded invocations in
    ``results/`` keep working. ``o0``, ``o1``, ``o2`` are what the directories say
    now, and generalise past two objectives.
    """
    cleaned = prefix.strip().rstrip("_")
    if cleaned in ("v", "c"):
        return 1 if cleaned == "v" else 0
    match = re.fullmatch(r"o?(\d)", cleaned)
    if match is None:
        raise ValueError(f"{prefix!r} does not name a sweep; expected 'o0', 'o1', ... (or legacy 'v'/'c')")
    return int(match.group(1))
