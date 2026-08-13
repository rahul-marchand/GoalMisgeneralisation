"""The header every measurement script writes above its output.

A file in ``results/`` is the verbatim output of one script against one set of
checkpoints, and the value of keeping it is that a number in a figure can be
traced back to the run that produced it. That only works if the file says what
produced it.

The header has recorded the commit and the arguments since early on, and both
are genuinely useful — the commit pins the code, and the arguments pin which
agent and which levels. What it never recorded is **which script ran**, because
it printed ``sys.argv[1:]`` and dropped ``argv[0]``. So the producing script had
to be inferred from the shape of the flags: ``--sweep`` meaning ``018``,
``--extrapolate`` meaning ``014``. That is exactly the guessing the header
exists to prevent, and it is why ``results/README.md`` had to be maintained by
hand and fell 35 files behind.

Adding one line makes the index generable instead, on the same principle as
``MANIFEST.md``: written by a script, committed, and the diff is the record of
what changed.

The parser reads the old two-line form as well, so the files already on disk
keep whatever provenance they have rather than being reduced to "unknown".
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_LABEL = 7


def commit() -> str:
    """The short SHA, or ``unknown`` outside a repository.

    Public because ``012`` records it in the JSON the figures read, so the
    number in a figure and the number in its results file come from one place.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def header(argv: list[str] | None = None) -> str:
    """Three lines naming the script, the commit and the arguments.

    Returned rather than printed, and without a trailing newline, so each caller
    keeps whatever spacing it already had between the header and its first
    table.
    """
    argv = sys.argv if argv is None else argv
    script = Path(argv[0]).name if argv and argv[0] else "unknown"
    return f"{'script':<{_LABEL}}{script}\n" f"{'commit':<{_LABEL}}{commit()}\n" f"{'argv':<{_LABEL}}{' '.join(argv[1:])}"


@dataclass(frozen=True)
class Provenance:
    """What a results file says about where it came from."""

    script: str | None
    commit: str | None
    argv: str | None

    @property
    def complete(self) -> bool:
        return self.script is not None and self.commit is not None


def parse_header(text: str) -> Provenance:
    """Read a results file's header, tolerating the old form that had no script line.

    Only the first few lines are inspected: an argument value further down the
    file could otherwise be mistaken for a header field, and a results file is
    allowed to contain anything at all below its header.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines()[:4]:
        match = re.fullmatch(r"(script|commit|argv)\s+(.*)", line.rstrip())
        if match is None:
            if fields:
                break
            continue
        fields[match.group(1)] = match.group(2).strip()
    return Provenance(script=fields.get("script"), commit=fields.get("commit"), argv=fields.get("argv"))
