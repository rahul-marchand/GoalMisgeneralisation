"""Silence gymnasium's wrapper-deprecation warnings, which drown the results files.

A file in ``results/`` is the verbatim output of a measurement script, and the
value of keeping it is that a number in a figure can be read out of it. Before
this, 84% of that corpus was one of five gymnasium deprecation messages -- 7.0 MB
of the 8.4 MB, 44,818 lines of 64,226 -- printed once per vectorised environment
construction and interleaved *into* the tables, because the sweeps redirect
stderr into the same file.

Two things follow from that beyond the size. ``provenance.parse_header`` reads
only the first few lines of a file, so a header can be pushed out of reach by
the warnings that precede it; and ``early_warning_report`` recovers its numbers
from those tables with line-anchored regexes, which have to survive arbitrary
text arriving between the rows.

The filter matches on the deprecation phrasing rather than raising gymnasium's
log level, so a warning that is actually about *our* usage still gets through.
Both messages come from cleanba's code reaching through a wrapper, which we do
not edit; there is nothing to fix on our side and nothing lost by not seeing it.
"""

from __future__ import annotations

import warnings

_DEPRECATION = r".*deprecated and will be (replaced|removed)"


def install() -> None:
    """Idempotent, and safe to call after ``warnings`` has already been configured."""
    warnings.filterwarnings("ignore", message=_DEPRECATION, category=UserWarning)
