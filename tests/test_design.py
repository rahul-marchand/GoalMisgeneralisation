"""Tests for the arm allocation.

The design is the whole reason the campaign is worth running: the first grid's
problem was not too few arms but too little leverage, and a design that
quietly lost its range would reproduce that failure at four times the GPU cost
while looking like a bigger experiment.

So these assert the properties the leverage argument depends on — range,
symmetry, distinctness — rather than the exact offsets, which are a judgement
call and may be retuned.
"""

from __future__ import annotations

import pytest

from goalmisgen.design import (
    DEFAULT_OFFSETS,
    Arm,
    arm_values,
    check_no_preference_flip,
    leverage,
    sweep_arms,
)

BASE_TWO = (1.0, 0.5)


def offsets_of(arms: list[Arm]) -> list[float]:
    return [a.offset for a in arms]


def test_the_first_grid_is_the_baseline_being_beaten() -> None:
    """Seven arms over -0.2 to +0.4, which gave reliabilities of 0.14 to 0.29."""
    assert leverage([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.28)


def test_the_design_buys_an_order_of_magnitude_more_leverage() -> None:
    """The claim in the module docstring, asserted rather than asserted-in-prose."""
    arms = sweep_arms(objective=1)

    assert leverage(offsets_of(arms)) == pytest.approx(3.049)
    assert leverage(offsets_of(arms)) > 10 * 0.28


def test_the_design_is_the_size_it_claims() -> None:
    assert len(sweep_arms(objective=1)) == 25


def test_offsets_are_symmetric_so_drift_cannot_leak_into_the_axis() -> None:
    """The first grid's asymmetry produced a confident-looking null."""
    offsets = offsets_of(sweep_arms(objective=1))

    assert sum(offsets) == pytest.approx(0.0)
    for offset in offsets:
        assert offsets.count(offset) == offsets.count(-offset)


def test_every_offset_is_distinct_so_no_arm_replaces_another() -> None:
    """``014`` keys arms by the value they trained at; a repeat would collide."""
    offsets = offsets_of(sweep_arms(objective=1))
    assert len(set(offsets)) == len(offsets)


def test_a_repeated_offset_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(ValueError, match="distinct"):
        sweep_arms(objective=1, offsets=(0.45, 0.45, 0.20))


def test_the_negative_side_is_generated_not_given() -> None:
    with pytest.raises(ValueError, match="positive side only"):
        sweep_arms(objective=1, offsets=(0.45, -0.45))


def test_most_of_the_leverage_sits_in_the_extreme_band() -> None:
    """Contribution goes as the square of the offset, so range beats resolution."""
    offsets = offsets_of(sweep_arms(objective=1))
    widest = max(abs(o) for o in offsets)
    band = [o for o in offsets if abs(o) >= widest - 0.08]

    assert leverage(band) / leverage(offsets) > 0.85


def test_interior_offsets_survive_for_the_linearity_check() -> None:
    """A design confined to one band cannot detect that the map is curved."""
    magnitudes = {abs(o) for o in offsets_of(sweep_arms(objective=1)) if o != 0}
    assert len([m for m in magnitudes if m < 0.35]) >= 4


def test_there_is_a_null_arm_to_measure_drift() -> None:
    assert sum(1 for a in sweep_arms(objective=1) if a.offset == 0.0) == 1


def test_every_arm_gets_its_own_seed() -> None:
    arms = sweep_arms(objective=1)
    assert len({a.seed for a in arms}) == len(arms)


def test_the_widest_arms_come_first() -> None:
    """An interrupted sweep should degrade to a smaller version of itself."""
    magnitudes = [abs(a.offset) for a in sweep_arms(objective=1)]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_arm_values_move_only_the_swept_objective() -> None:
    arm = Arm(sweep="o1", objective=1, offset=0.45, seed=1)
    assert arm_values(BASE_TWO, arm) == (1.0, 0.95)

    arm = Arm(sweep="o0", objective=0, offset=-0.45, seed=1)
    assert arm_values(BASE_TWO, arm) == (0.55, 0.5)


def test_neither_sweep_reverses_which_objective_is_worth_more() -> None:
    """Past the flip the arm learns the opposite preference, not more of the same one."""
    for objective in (0, 1):
        assert check_no_preference_flip(BASE_TWO, sweep_arms(objective=objective)) == []


def test_a_flip_is_reported_rather_than_trained() -> None:
    too_far = [Arm(sweep="o1", objective=1, offset=0.6, seed=1)]
    problems = check_no_preference_flip(BASE_TWO, too_far)

    assert len(problems) == 1
    assert "reordering" in problems[0]


def test_the_design_generalises_to_three_objectives() -> None:
    base = (1.0, 0.55, 0.4)
    # Objective 2 sits closest to its neighbour, so its usable range is narrowest.
    narrow = (0.14, 0.12, 0.10, 0.05)
    arms = sweep_arms(objective=2, offsets=narrow)

    assert check_no_preference_flip(base, arms) == []
    assert len(arms) == 9


def test_every_default_offset_appears_on_both_sides() -> None:
    offsets = set(offsets_of(sweep_arms(objective=1)))
    for magnitude in DEFAULT_OFFSETS:
        assert magnitude in offsets and -magnitude in offsets
