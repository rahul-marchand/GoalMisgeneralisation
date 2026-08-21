"""Tests for the results-file header.

The header is what lets a number in a figure be traced to the run that produced
it, and it spent most of the project unable to say which script ran — it printed
``sys.argv[1:]`` and dropped ``argv[0]``. So the tests that matter are that the
script name is there now, and that the parser still reads the files written
before it was, since those are most of ``results/``.
"""

from __future__ import annotations

from goalmisgen.provenance import Provenance, header, parse_header

OLD_FORM = """commit f752073
argv   --base /workspace/data/runs/novalue11/local-files/cp_140206080 --sweep v 0.5

  channel  enrichment
  ch07     2.29
"""


def test_the_header_names_the_script_that_wrote_it() -> None:
    text = header(["experiments/018_which_channels.py", "--base", "cp_1"])
    assert text.splitlines()[0] == "script 018_which_channels.py"


def test_the_header_carries_the_arguments() -> None:
    text = header(["experiments/014_value_axis_analysis.py", "--at", "-1", "--leave-one-out"])
    assert "argv   --at -1 --leave-one-out" in text


def test_the_header_has_no_trailing_newline() -> None:
    """Each caller keeps whatever spacing it already had below the header."""
    assert not header(["x.py"]).endswith("\n")


def test_a_header_round_trips_through_the_parser() -> None:
    parsed = parse_header(header(["experiments/021_own_task.py", "--arms", "/data/runs"]))

    assert parsed.script == "021_own_task.py"
    assert parsed.argv == "--arms /data/runs"
    assert parsed.complete


def test_the_old_two_line_form_still_parses() -> None:
    """Most of results/ was written before the script line existed."""
    parsed = parse_header(OLD_FORM)

    assert parsed.script is None
    assert parsed.commit == "f752073"
    assert parsed.argv.startswith("--base /workspace/data/runs/novalue11")
    assert not parsed.complete


def test_a_file_with_no_header_at_all_parses_as_empty() -> None:
    """seed-comparison.txt is a hand-written summary and has never had one."""
    parsed = parse_header("Two seeds of the same experiment, compared like for like.\n\nnovalue11 ...")

    assert parsed == Provenance(script=None, commit=None, argv=None)
    assert not parsed.complete


def test_the_body_cannot_be_mistaken_for_a_header() -> None:
    """An argument value further down must not be read as a header field."""
    body = "script real.py\ncommit abc1234\nargv   --x 1\n\nsome table\ncommit deadbee\n"
    assert parse_header(body).commit == "abc1234"


def test_a_missing_argv_zero_does_not_crash() -> None:
    assert parse_header(header([])).script == "unknown"
