"""The rule that decides whether a rung of the ladder has an axis.

Kept apart from the script that uses it because it is the load-bearing judgement
of the whole ladder: it is what turns a table of numbers into "the axis appears
at 20M". A rule that lives inside a print loop cannot be argued with.
"""

from __future__ import annotations

from goalmisgen.analysis.behaviour import write_verdict


def point(offset: float, value: float, half_width: float = 0.5, reached: float = 1.0) -> dict:
    return {"offset": offset, "point": value, "low": value - half_width, "high": value + half_width, "reached": reached}


def test_disjoint_intervals_at_the_extremes_are_a_write() -> None:
    result = write_verdict(1.0, [point(-0.45, 2.0), point(0.45, 9.0)])

    assert result.works
    assert result.moved == 7.0


def test_overlapping_intervals_are_not() -> None:
    """Means that differ are not enough; they have to differ by more than the noise."""
    result = write_verdict(1.0, [point(-0.45, 7.0, half_width=1.0), point(0.45, 7.4, half_width=1.0)])

    assert not result.works
    assert result.verdict == "no axis"


def test_the_direction_of_the_move_is_not_prejudged() -> None:
    """Colour 0's axis moves the rate the opposite way to colour 1's."""
    assert write_verdict(1.0, [point(-0.45, 9.0), point(0.45, 2.0)]).works


def test_an_incompetent_base_is_reported_apart_from_a_missing_axis() -> None:
    """Early rungs are expected to be here, and it is not evidence about the axis.

    Reading it as "no axis" would date the axis to whenever the agent became
    competent, whenever the axis actually arrived.
    """
    result = write_verdict(0.40, [point(-0.45, 2.0), point(0.45, 9.0)])

    assert result.verdict == "base cannot do the task"
    assert not result.works


def test_writes_that_break_the_agent_are_dropped_before_judging() -> None:
    """An exchange rate read off episodes the agent did not finish is not a rate."""
    written = [point(-0.45, 2.0, reached=0.30), point(-0.20, 6.9), point(0.20, 7.1)]
    result = write_verdict(1.0, written)

    assert result.usable == 2
    assert not result.works  # the two survivors overlap


def test_one_usable_point_cannot_decide_anything() -> None:
    result = write_verdict(1.0, [point(-0.45, 2.0, reached=0.1), point(0.45, 9.0)])

    assert result.usable == 1
    assert result.verdict == "no axis"
