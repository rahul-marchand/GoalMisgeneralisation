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


def test_a_binding_reach_floor_is_reported_rather_than_buried() -> None:
    """The 70.1M rung of novalue11.s1234, which the floor decided on its own.

    The writes were graded and their extreme intervals disjoint; base reach was
    94.0% against a floor of 95% fixed before any data existed. The verdict stays
    conservative -- relaxing a preregistered threshold after seeing the number is
    the thing preregistration exists to stop -- but it must not swallow the
    evidence, or a one-point miss silently picks the headline answer.
    """
    written = [
        point(-0.45, 6.6, half_width=1.1, reached=0.903),
        point(-0.20, 4.8, half_width=1.0, reached=0.924),
        point(0.20, 2.9, half_width=0.9, reached=0.952),
        point(0.45, 1.9, half_width=0.9, reached=0.956),
    ]

    result = write_verdict(0.940, written, reach_floor=0.95)

    assert result.verdict == "base cannot do the task"
    assert not result.works
    assert result.disjoint_ignoring_reach
    assert result.floor_is_binding
    assert result.min_reach == 0.903


def test_a_rung_that_genuinely_has_nothing_is_not_flagged_as_a_near_miss() -> None:
    """floor_is_binding must not fire just because the base was incompetent."""
    written = [point(-0.45, 0.2, half_width=2.0, reached=0.10), point(0.45, 0.3, half_width=2.0, reached=0.11)]

    result = write_verdict(0.06, written, reach_floor=0.95)

    assert not result.disjoint_ignoring_reach
    assert not result.floor_is_binding
