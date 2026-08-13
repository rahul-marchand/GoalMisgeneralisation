"""Tests for the one-off migration of the data volume.

This script runs once, against a rented volume holding every checkpoint the
project has produced, and there is no second copy. So the properties worth
testing are not that it renames things correctly on a happy path but that it
*refuses* in every case where it might rename something wrongly: an arm whose
saved values disagree with its old name, two things wanting one destination, a
run whose configuration cannot be read.

The fixture builds a miniature volume with the same shape as the real one —
including the parts that motivated the rename, namely ``threeobj2`` being
arms of ``threeobj`` rather than a separate experiment, and two datasets at the
same values but different level counts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "migrate_volume", Path(__file__).resolve().parents[1] / "scripts" / "migrate_volume.py"
)
assert _spec is not None and _spec.loader is not None
migrate = importlib.util.module_from_spec(_spec)
# ``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``, so
# the module has to be registered before it is executed rather than after.
sys.modules[_spec.name] = migrate
_spec.loader.exec_module(migrate)


def write_run(root: Path, relative: str, values: list[float], steps: int, *, arm: bool = False) -> Path:
    """A run directory shaped like the ones on the volume, config and all."""
    run = root / relative
    payload = {
        "cfg": {
            "total_timesteps": steps,
            "train_env": {"objective_values": values, "n_objectives": len(values), "level_dataset": "/levels/x"},
        }
    }
    if arm:
        (run / "init").mkdir(parents=True)
        (run / "init" / "cfg.json").write_text(json.dumps(payload))
    else:
        checkpoint = run / "local-files" / f"cp_{steps}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "cfg.json").write_text(json.dumps(payload))
    (run / "local-files").mkdir(parents=True, exist_ok=True)
    (run / "local-files" / "model").write_bytes(b"weights")
    return run


def write_dataset(root: Path, relative: str, values: list[float] | None, n_levels: int) -> Path:
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(json.dumps({"n_levels": n_levels, "fingerprint": "abc123"}))
    stored = np.tile(np.array(values, dtype=np.float64), (4, 1)) if values else np.arange(8.0).reshape(4, 2)
    np.save(directory / "values.npy", stored)
    return directory


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    write_run(tmp_path, "runs/novalue11", [1.0, 0.5], 150_000_000)
    write_run(tmp_path, "runs/novalue11_seed5678", [1.0, 0.5], 150_000_000)
    write_run(tmp_path, "threeobj/runs/base", [1.0, 0.65, 0.3], 80_000_000)

    # Colour 1 swept up, colour 0 swept up, and the null arm that measures drift.
    write_run(tmp_path, "valueaxis/runs/v070", [1.0, 0.7], 3_000_000, arm=True)
    write_run(tmp_path, "valueaxis/runs/v050", [1.0, 0.5], 3_000_000, arm=True)
    write_run(tmp_path, "valueaxis/runs/c120", [1.2, 0.5], 3_000_000, arm=True)
    write_run(tmp_path, "valueaxis_s5678/runs/v070", [1.0, 0.7], 750_000, arm=True)

    # The point of the exercise: threeobj2 is the same grid at wider offsets.
    write_run(tmp_path, "threeobj/runs/o1_045", [1.0, 0.45, 0.3], 1_000_000, arm=True)
    write_run(tmp_path, "threeobj/runs/m_120_045_030", [1.2, 0.45, 0.3], 1_000_000, arm=True)
    write_run(tmp_path, "threeobj2/runs/o1_025", [1.0, 0.25, 0.3], 3_000_000, arm=True)

    write_dataset(tmp_path, "levels11", [1.0, 0.5], 1_000_000)
    write_dataset(tmp_path, "valueaxis/levels/v050", [1.0, 0.5], 500_000)
    write_dataset(tmp_path, "valueaxis/levels/v070", [1.0, 0.7], 500_000)
    write_dataset(tmp_path, "levels11rv", None, 1_000_000)
    return tmp_path


def destinations(plan, root: Path) -> dict[str, str]:
    return {str(m.source.relative_to(root)): str(m.destination.relative_to(root)) for m in plan.moves}


def test_arms_are_named_from_what_they_trained_on(volume: Path) -> None:
    """v070 moved objective 1 up by 0.2 from a base of 0.5, so it is o1+020."""
    moves = destinations(migrate.plan_runs(volume), volume)

    assert moves["valueaxis/runs/v070"] == "runs/novalue11.s1234/arms/o1+020@3M"
    assert moves["valueaxis/runs/c120"] == "runs/novalue11.s1234/arms/o0+020@3M"
    assert moves["threeobj/runs/o1_045"] == "runs/threeobj.even.s1234/arms/o1-020@1M"


def test_the_null_arm_is_named_by_its_zero_offset(volume: Path) -> None:
    """v050 is colour 1's null arm, so it keeps colour 1's sweep and a zero offset."""
    moves = destinations(migrate.plan_runs(volume), volume)
    assert moves["valueaxis/runs/v050"] == "runs/novalue11.s1234/arms/o1+000@3M"


def test_arm_length_lands_in_the_path(volume: Path) -> None:
    """The 3M-versus-750k incomparability becomes visible instead of documented."""
    moves = destinations(migrate.plan_runs(volume), volume)

    assert moves["valueaxis/runs/v070"].endswith("@3M")
    assert moves["valueaxis_s5678/runs/v070"].endswith("@750k")
    assert "novalue11.s1234" in moves["valueaxis/runs/v070"]
    assert "novalue11.s5678" in moves["valueaxis_s5678/runs/v070"]


def test_threeobj2_becomes_arms_of_threeobj_even(volume: Path) -> None:
    """It was never a separate experiment — same base, wider offsets, longer arms."""
    moves = destinations(migrate.plan_runs(volume), volume)
    assert moves["threeobj2/runs/o1_025"] == "runs/threeobj.even.s1234/arms/o1-040@3M"


def test_composition_arms_keep_their_values_rather_than_an_offset(volume: Path) -> None:
    moves = destinations(migrate.plan_runs(volume), volume)
    assert moves["threeobj/runs/m_120_045_030"] == "runs/threeobj.even.s1234/arms/m_120_045_030@1M"


def test_datasets_at_the_same_values_and_size_collapse_but_different_sizes_do_not(volume: Path) -> None:
    moves = destinations(migrate.plan_datasets(volume), volume)

    assert moves["levels11"] == "levels/values/1.00-0.50@1M"
    assert moves["valueaxis/levels/v050"] == "levels/values/1.00-0.50@500k"
    assert moves["valueaxis/levels/v070"] == "levels/values/1.00-0.70@500k"


def test_randomised_value_datasets_keep_their_own_name(volume: Path) -> None:
    """They have no fixed tuple to be keyed by, so the library cannot hold them."""
    moves = destinations(migrate.plan_datasets(volume), volume)
    assert moves["levels11rv"] == "levels/randomised/levels11rv"


def test_identical_datasets_collapse_to_one_copy(volume: Path) -> None:
    """Several campaigns generated their own 500k at (1.0, 0.5); one should serve all."""
    write_dataset(volume, "valueaxis_s5678/levels/v050", [1.0, 0.5], 500_000)

    plan = migrate.plan_datasets(volume)
    moved = destinations(plan, volume)

    assert plan.collisions() == {}
    assert sum(1 for d in moved.values() if d.endswith("1.00-0.50@500k")) == 1
    assert [str(s.relative_to(volume)) for s, _ in plan.duplicates] == ["valueaxis_s5678/levels/v050"]


def test_datasets_sharing_a_key_with_different_contents_stop_the_migration(volume: Path) -> None:
    """Split membership lives in the dataset, so a different holdout is a different dataset.

    Collapsing these would let an agent be evaluated on levels it trained on.
    """
    other = write_dataset(volume, "valueaxis_s5678/levels/v050", [1.0, 0.5], 500_000)
    np.save(other / "split_train.npy", np.arange(7))

    plan = migrate.plan_datasets(volume)

    assert plan.duplicates == []
    assert any("different contents" in reason for _, reason in plan.unaccounted)
    assert sum(1 for d in destinations(plan, volume).values() if d.endswith("1.00-0.50@500k")) == 1


def test_duplicates_are_retired_rather_than_left_lying_around(volume: Path) -> None:
    write_dataset(volume, "valueaxis_s5678/levels/v050", [1.0, 0.5], 500_000)
    plan = migrate.plan_datasets(volume)
    record = volume / "MIGRATION.json"
    migrate.apply(plan, record)

    migrate.retire(record, volume)

    assert not (volume / "valueaxis_s5678" / "levels" / "v050").exists()
    assert (volume / "retired" / "valueaxis_s5678" / "levels" / "v050" / "meta.json").is_file()


def test_a_run_named_in_the_table_but_absent_is_reported_not_skipped(volume: Path) -> None:
    plan = migrate.plan_runs(volume)
    absent = {str(path) for path, _ in plan.unaccounted}
    assert "runs/maze11" in absent
    assert "threeobj_v2/runs/base" in absent


def test_a_run_whose_config_cannot_be_read_is_reported(volume: Path) -> None:
    (volume / "valueaxis" / "runs" / "broken" / "local-files").mkdir(parents=True)
    (volume / "valueaxis" / "runs" / "broken" / "init").mkdir()
    (volume / "valueaxis" / "runs" / "broken" / "init" / "cfg.json").write_text("{not json")

    plan = migrate.plan_runs(volume)

    assert any("broken" in str(path) for path, _ in plan.unaccounted)
    assert not any("broken" in str(m.source) for m in plan.moves)


def test_an_arm_with_the_wrong_objective_count_is_reported_not_renamed(volume: Path) -> None:
    """A two-objective arm under a three-objective base is a filing error, not an offset."""
    write_run(volume, "threeobj/runs/wrong", [1.0, 0.5], 1_000_000, arm=True)

    plan = migrate.plan_runs(volume)

    assert any("wrong" in str(path) and "objectives" in reason for path, reason in plan.unaccounted)


def test_two_things_wanting_one_destination_are_caught(volume: Path) -> None:
    """Duplicating an arm's values under one base must not silently overwrite."""
    write_run(volume, "valueaxis/runs/v070_again", [1.0, 0.7], 3_000_000, arm=True)

    plan = migrate.plan_runs(volume)

    assert plan.collisions()


def test_apply_then_verify_round_trips(volume: Path) -> None:
    plan = migrate.plan_runs(volume)
    record = volume / "MIGRATION.json"

    migrate.apply(plan, record)

    assert migrate.verify(record) == 0
    assert (volume / "runs" / "novalue11.s1234" / "arms" / "o1+020@3M" / "local-files" / "model").is_file()


def test_an_agent_verifies_despite_the_arms_nested_inside_it(volume: Path) -> None:
    """Arms live under the agent they were fine-tuned from, so the new agent
    directory holds strictly more than the old one did. Verification has to
    compare like with like or every agent reads as corrupted."""
    plan = migrate.plan_runs(volume)
    record = volume / "MIGRATION.json"
    migrate.apply(plan, record)

    agent = volume / "runs" / "novalue11.s1234"
    assert (agent / "arms" / "o1+020@3M").is_dir(), "the arm should be nested inside its agent"
    assert migrate.verify(record) == 0


def test_verify_catches_a_partial_copy(volume: Path) -> None:
    """The failure being guarded against: a truncated checkpoint looks normal."""
    plan = migrate.plan_runs(volume)
    record = volume / "MIGRATION.json"
    migrate.apply(plan, record)

    (volume / "runs" / "novalue11.s1234" / "arms" / "o1+020@3M" / "local-files" / "model").write_bytes(b"trunc")

    assert migrate.verify(record) == 1


def test_base_json_points_at_the_newest_checkpoint(volume: Path) -> None:
    """The rule is stable even where the number is surprising — threeobj.even's
    canonical checkpoint is cp_70103040, not the 80M it trained for."""
    plan = migrate.plan_runs(volume)
    record = volume / "MIGRATION.json"
    migrate.apply(plan, record)

    migrate.finalise(volume)

    payload = json.loads((volume / "runs" / "novalue11.s1234" / "BASE.json").read_text())
    assert payload["checkpoint"] == "local-files/cp_150000000"
    assert payload["values"] == [1.0, 0.5]


def test_base_json_is_written_after_verification_not_before(volume: Path) -> None:
    """It exists only on the new side, so writing it early would fail the checksum."""
    plan = migrate.plan_runs(volume)
    record = volume / "MIGRATION.json"
    migrate.apply(plan, record)

    assert migrate.verify(record) == 0
    migrate.finalise(volume)
    assert (volume / "runs" / "novalue11.s1234" / "BASE.json").is_file()


def test_retire_moves_rather_than_deletes(volume: Path) -> None:
    plan = migrate.plan_runs(volume)
    record = volume / "MIGRATION.json"
    migrate.apply(plan, record)

    migrate.retire(record, volume)

    assert not (volume / "valueaxis" / "runs" / "v070").exists()
    assert (volume / "retired" / "valueaxis" / "runs" / "v070" / "local-files" / "model").is_file()


def test_each_sweeps_null_arm_survives_the_move(volume: Path) -> None:
    """v050 and c100 are both trained at the base values, so nothing in their
    configuration says which sweep's drift they measure. Naming both from the
    values alone collapsed them onto one path and would have thrown one away."""
    write_run(volume, "valueaxis/runs/c100", [1.0, 0.5], 3_000_000, arm=True)

    plan = migrate.plan_runs(volume)
    moves = destinations(plan, volume)

    assert plan.collisions() == {}
    assert moves["valueaxis/runs/v050"] == "runs/novalue11.s1234/arms/o1+000@3M"
    assert moves["valueaxis/runs/c100"] == "runs/novalue11.s1234/arms/o0+000@3M"


def test_old_sweep_prefixes_map_to_the_objective_they_moved() -> None:
    assert migrate.sweep_objective("v050") == 1
    assert migrate.sweep_objective("c100") == 0
    assert migrate.sweep_objective("o2_040") == 2
    assert migrate.sweep_objective("m_120_045_030") is None


def test_a_null_arm_whose_sweep_cannot_be_told_is_refused() -> None:
    """Better to report it than to guess which drift measurement it is."""
    with pytest.raises(ValueError, match="does not say which sweep"):
        migrate.arm_name((1.0, 0.5), (1.0, 0.5), 750_000, None)
