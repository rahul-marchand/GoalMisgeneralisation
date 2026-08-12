"""Pull the value-axis numbers out of the results files into figures/data.

    uv run python figures/extract_value_axis.py

``make_figures.py`` reads JSON and never has a number typed into it, so that a
re-measurement cannot leave a stale annotation behind on a plot. The experiment
scripts print tables rather than emitting JSON, so this parses those tables once
and writes the result where the figures can find it.

Parsing formatted text is fragile, which is why every extraction here asserts on
the number of rows it found. A changed table format fails loudly instead of
silently producing a figure with three points on it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).parent / "data"


def rows(text: str, pattern: str, expected: int, what: str) -> list[tuple[float, ...]]:
    found = [tuple(float(g) for g in m.groups()) for m in re.finditer(pattern, text, re.MULTILINE)]
    if len(found) != expected:
        raise SystemExit(f"expected {expected} {what}, parsed {len(found)} — has the table format changed?")
    return found


def main() -> None:
    OUT.mkdir(exist_ok=True)
    full = (RESULTS / "value-axis-full.txt").read_text()
    held = (RESULTS / "value-axis-heldout.txt").read_text()

    # The arms themselves, and the base they were all fine-tuned from.
    trained = rows(full, r"v=(\d\.\d\d) fine-tuned\s+(-?\d+\.\d)", 7, "fine-tuned arms")
    base = rows(full, r"base, untouched\s+(-?\d+\.\d)", 1, "base measurement")[0][0]

    # Written from an axis fitted *without* the arm being predicted. This is the
    # claim; the in-sample version cannot separate an axis from a lookup of the
    # arms it was built from.
    heldout = rows(held, r"^\s+(\d\.\d\d)\s+(-?\d+\.\d)\s+(-?\d+\.\d)\s+[+-]\d+\.\d", 6, "held-out writes")

    # Values outside the fitted grid. These come from the full-data axis, since
    # there is no arm at 0.20 or 1.10 to hold out.
    beyond = rows(full, r"v=(\d\.\d\d) written \(unseen\)\s+(-?\d+\.\d)", 2, "extrapolations")

    # Directions of the same length, chosen at random. The control that makes
    # the rest mean anything.
    random = rows(full, r"random, matched to (\d\.\d)\s+(-?\d+\.\d)", 2, "random controls")

    payload = {
        "base_value": 0.5,
        "step_penalty": 0.05,
        "other_objective": 1.0,
        "base_exchange_rate": base,
        "trained": [{"value": v, "steps": s} for v, s in trained],
        "written_heldout": [{"value": v, "trained": t, "written": w} for v, t, w in heldout],
        "written_beyond_grid": [{"value": v, "steps": s} for v, s in beyond],
        "random_controls": [{"magnitude": m, "steps": s} for m, s in random],
    }
    (OUT / "value_axis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT / 'value_axis.json'}")
    print(f"  base {base} steps, {len(trained)} arms, {len(heldout)} held-out writes, "
          f"{len(beyond)} beyond the grid, {len(random)} controls")


if __name__ == "__main__":
    main()
