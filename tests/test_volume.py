"""Tests for the naming scheme on the data volume.

These names are load-bearing. An arm's directory is what tells the analysis
which agent it belongs to and how long it trained, and the reason the scheme
exists is that both facts used to live in prose — where the 3M-versus-750k
incomparability between the two seeds sat as a paragraph in
``results/seed-comparison.txt`` rather than as something a script could see.

So the cases that matter are the ones where a name would be *accepted while
meaning something other than it says*: a step count that rounds, an offset finer
than the name can carry, a missing sign. Those must raise rather than round.
"""

from __future__ import annotations

import pytest

from goalmisgen.volume import (
    ArmName,
    arm_dirname,
    arm_lengths,
    composition_arm_dirname,
    dataset_dirname,
    discover_arms,
    is_composition_arm,
    offset_tag,
    parse_arm_dirname,
    parse_dataset_dirname,
    parse_steps_tag,
    parse_values_tag,
    steps_tag,
    sweep_index,
    values_tag,
)


def test_arm_name_round_trips() -> None:
    for sweep, offset, steps in [("c1", 0.2, 750_000), ("c0", -0.05, 3_000_000), ("o2", 0.0, 1_000_000)]:
        name = arm_dirname(sweep, offset, steps)
        assert parse_arm_dirname(name) == ArmName(sweep=sweep, offset=offset, steps=steps)


def test_names_look_the_way_the_scheme_says() -> None:
    assert arm_dirname("c1", 0.2, 750_000) == "c1+020@750k"
    assert arm_dirname("c0", -0.05, 3_000_000) == "c0-005@3M"
    assert arm_dirname("c1", 0.0, 1_000_000) == "c1+000@1M"


def test_the_null_arm_announces_itself() -> None:
    """It measures drift rather than value, so every fit has to hold it out."""
    assert parse_arm_dirname("c1+000@750k").is_null
    assert not parse_arm_dirname("c1+010@750k").is_null


def test_offsets_order_by_parsing_not_by_filename() -> None:
    """``ls`` puts '+' before '-', so anything presenting arms in offset order must parse.

    Nothing depends on lexical order — ``014``'s ``arm_checkpoints`` keys by the
    parsed value and sorts that — but a tool that sorted the directory listing
    and believed it would put every negative offset after every positive one.
    """
    names = [arm_dirname("c1", o, 750_000) for o in (0.4, -0.2, 0.0, 0.1)]

    assert sorted(names) == ["c1+000@750k", "c1+010@750k", "c1+040@750k", "c1-020@750k"]
    assert [parse_arm_dirname(n).offset for n in sorted(names, key=lambda n: parse_arm_dirname(n).offset)] == [
        -0.2,
        0.0,
        0.1,
        0.4,
    ]


def test_a_step_count_that_would_round_is_refused() -> None:
    """Two arms claiming the same length must actually have it."""
    with pytest.raises(ValueError, match="no exact tag"):
        steps_tag(1_234_567)


def test_an_offset_finer_than_the_name_is_refused() -> None:
    with pytest.raises(ValueError, match="finer than"):
        offset_tag(0.125)


def test_step_tags_round_trip() -> None:
    for steps in (750_000, 1_000_000, 3_000_000, 187_000):
        assert parse_steps_tag(steps_tag(steps)) == steps


@pytest.mark.parametrize("name", ["v070", "c1+020", "c1020@750k", "c1+20@750k", "c1+020@750000", "", "arms"])
def test_things_that_are_not_arms_parse_as_none(name: str) -> None:
    """Callers walk a directory holding other things; skipping beats raising."""
    assert parse_arm_dirname(name) is None


def test_composition_arms_are_recognised_but_not_arms() -> None:
    """They move several values at once, so they have no single offset to fit."""
    name = composition_arm_dirname([1.2, 0.45, 0.3], 1_000_000)
    assert name == "m_120_045_030@1M"
    assert is_composition_arm(name)
    assert parse_arm_dirname(name) is None


def test_values_tag_round_trips() -> None:
    for values in [(1.0, 0.5), (1.0, 0.65, 0.3), (1.0, 0.55, 0.4)]:
        assert parse_values_tag(values_tag(values)) == values


def test_the_same_values_give_the_same_dataset_key() -> None:
    """One dataset serves every seed, so the key must not carry who asked for it."""
    assert values_tag([1.0, 0.5]) == values_tag((1.00, 0.500))


def test_a_negative_objective_value_is_refused() -> None:
    """Four tenths below 0.3 is a punishment, i.e. a different task, not a wider grid."""
    with pytest.raises(ValueError, match="cannot be negative"):
        values_tag([1.0, -0.1])


def test_datasets_at_the_same_values_but_different_sizes_stay_apart() -> None:
    """levels11 is 1M at (1, 0.5); valueaxis/levels/v050 is 500k at the same values.

    Layouts are a function of the seed and the count, so the smaller is not a
    prefix of the larger and merging them would retrain runs on different mazes.
    """
    assert dataset_dirname([1.0, 0.5], 1_000_000) == "1.00-0.50@1M"
    assert dataset_dirname([1.0, 0.5], 500_000) == "1.00-0.50@500k"
    assert dataset_dirname([1.0, 0.5], 1_000_000) != dataset_dirname([1.0, 0.5], 500_000)


def test_dataset_names_round_trip() -> None:
    for values, count in [((1.0, 0.5), 1_000_000), ((1.0, 0.65, 0.3), 150_000)]:
        assert parse_dataset_dirname(dataset_dirname(values, count)) == (values, count)


def test_a_dataset_without_its_level_count_is_refused() -> None:
    with pytest.raises(ValueError, match="missing its level count"):
        parse_dataset_dirname("1.00-0.50")


def make_arm(root, name: str, checkpoints: int = 1):
    arm = root / name / "local-files"
    arm.mkdir(parents=True)
    for i in range(checkpoints):
        (arm / f"cp_{(i + 1) * 1000}").mkdir()
    return arm


def test_discover_arms_keys_by_the_value_each_arm_trained_at(tmp_path) -> None:
    """The analysis regresses against the value, so renaming did not change its shape."""
    for name in ("o1+020@750k", "o1-020@750k", "o1+000@750k"):
        make_arm(tmp_path, name)

    found = discover_arms(tmp_path, objective=1, base_value=0.5, steps=750_000)

    assert sorted(found) == [0.3, 0.5, 0.7]


def test_discover_arms_ignores_the_other_sweep(tmp_path) -> None:
    make_arm(tmp_path, "o1+020@750k")
    make_arm(tmp_path, "o0+020@750k")

    assert sorted(discover_arms(tmp_path, 1, 0.5, steps=750_000)) == [0.7]
    assert sorted(discover_arms(tmp_path, 0, 1.0, steps=750_000)) == [1.2]


def test_mixing_arm_lengths_is_refused_rather_than_reported(tmp_path) -> None:
    """An agent's arms/ now holds every sweep ever run against it."""
    make_arm(tmp_path, "o1+020@750k")
    make_arm(tmp_path, "o1+020@3M")

    with pytest.raises(ValueError, match="holds arms at"):
        discover_arms(tmp_path, 1, 0.5)


def test_asking_for_one_length_selects_only_that_sweep(tmp_path) -> None:
    make_arm(tmp_path, "o1+020@750k")
    make_arm(tmp_path, "o1+040@3M")

    assert sorted(discover_arms(tmp_path, 1, 0.5, steps=750_000)) == [0.7]
    assert sorted(discover_arms(tmp_path, 1, 0.5, steps=3_000_000)) == [0.9]
    assert arm_lengths(tmp_path) == {750_000, 3_000_000}


def test_discover_arms_picks_the_requested_checkpoint(tmp_path) -> None:
    make_arm(tmp_path, "o1+020@750k", checkpoints=4)

    assert discover_arms(tmp_path, 1, 0.5, steps=750_000, at=0)[0.7].name == "cp_1000"
    assert discover_arms(tmp_path, 1, 0.5, steps=750_000, at=-1)[0.7].name == "cp_4000"


def test_an_arm_with_no_checkpoints_is_skipped(tmp_path) -> None:
    (tmp_path / "o1+020@750k").mkdir(parents=True)
    assert discover_arms(tmp_path, 1, 0.5, steps=750_000) == {}


def test_legacy_sweep_prefixes_still_resolve() -> None:
    """Invocations recorded in results/ named the sweeps v and c."""
    assert sweep_index("v") == 1
    assert sweep_index("c") == 0
    assert sweep_index("o2") == 2


def test_an_unrecognised_sweep_name_is_refused() -> None:
    with pytest.raises(ValueError, match="does not name a sweep"):
        sweep_index("colour1")
