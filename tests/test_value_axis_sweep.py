"""Tests for the sweep driver's planning.

The driver spends hours of GPU per invocation, so what is worth testing is the
part that happens before it commits: which arms it decides to train, which
datasets it decides they need, and whether it stops when the grid has walked
outside the task.

Nothing here runs ``013`` or ``generate_levels``; ``--dry-run`` is the surface
under test, on the same principle as ``tests/test_value_axis_cli.py`` — the
entry point gets tested, not just the function behind it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("value_axis_sweep", REPO / "scripts" / "value_axis_sweep.py")
assert _spec is not None and _spec.loader is not None
sweep = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sweep
_spec.loader.exec_module(sweep)


def make_agent(root: Path, tag: str, values: list[float], checkpoint: str = "local-files/cp_140206080") -> Path:
    agent = root / "runs" / tag
    (agent / checkpoint).mkdir(parents=True)
    (agent / "BASE.json").write_text(
        json.dumps({"checkpoint": checkpoint, "values": values, "objectives": len(values), "steps": 150_000_000})
    )
    return agent


def dry_run(root: Path, tag: str, *extra: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "value_axis_sweep.py"),
            "--data",
            str(root),
            "--agent",
            tag,
            "--dry-run",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    make_agent(tmp_path, "novalue11.s1234", [1.0, 0.5])
    return tmp_path


def test_it_plans_both_sweeps_by_default(volume: Path) -> None:
    """Sweeping only one objective cannot separate a value from the gap."""
    out = dry_run(volume, "novalue11.s1234")

    assert "objectives   [0, 1]" in out
    assert "arms         50 x 750,000 steps" in out


def test_it_reports_the_leverage_it_is_buying(volume: Path) -> None:
    out = dry_run(volume, "novalue11.s1234")
    assert "leverage 3.049" in out


def test_arms_are_named_by_sweep_offset_and_length(volume: Path) -> None:
    out = dry_run(volume, "novalue11.s1234")

    assert "o1+045@750k" in out
    assert "o0-045@750k" in out
    assert "o1+000@750k" in out


def test_datasets_are_keyed_by_values_and_size_not_by_the_arm(volume: Path) -> None:
    """One library entry per value tuple, so a second agent regenerates nothing."""
    out = dry_run(volume, "novalue11.s1234")

    assert "1.00-0.95@150k" in out
    assert "0.55-0.50@150k" in out


def test_the_two_sweeps_share_the_base_dataset(volume: Path) -> None:
    """Both null arms sit at the base values, so only one dataset is needed for them."""
    out = dry_run(volume, "novalue11.s1234")
    assert out.count("1.00-0.50@150k") == 1


def test_an_existing_dataset_is_reused_rather_than_regenerated(volume: Path) -> None:
    (volume / "levels" / "values" / "1.00-0.95@150k").mkdir(parents=True)

    out = dry_run(volume, "novalue11.s1234")

    assert "present  1.00-0.95@150k" in out
    assert "generate 1.00-0.95@150k" not in out


def test_a_finished_arm_is_skipped(volume: Path) -> None:
    """So an interrupted sweep can be re-run without repeating work.

    Completion is judged by the length reached, not by a checkpoint existing —
    see test_an_arm_that_stopped_early_is_retrained_not_counted.
    """
    done = volume / "runs" / "novalue11.s1234" / "arms" / "o1+045@750k" / "local-files" / "cp_748800"
    done.mkdir(parents=True)

    out = dry_run(volume, "novalue11.s1234")

    assert "o1+045@750k already complete" in out


def test_a_grid_that_would_reorder_the_objectives_stops_the_sweep(tmp_path: Path) -> None:
    """At base values 0.6 and 0.5 the default offsets walk straight past the flip."""
    make_agent(tmp_path, "narrow.s1", [0.6, 0.5])

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "value_axis_sweep.py"),
            "--data",
            str(tmp_path),
            "--agent",
            "narrow.s1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode != 0
    assert "reordering the objectives" in result.stdout
    assert "Narrow the offsets" in result.stderr


def test_a_missing_base_json_says_what_writes_it(tmp_path: Path) -> None:
    (tmp_path / "runs" / "nothing.s1").mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "value_axis_sweep.py"),
            "--data",
            str(tmp_path),
            "--agent",
            "nothing.s1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode != 0
    assert "migrate_volume.py --retire" in result.stderr


def test_a_base_json_pointing_at_a_missing_checkpoint_is_caught(tmp_path: Path) -> None:
    agent = tmp_path / "runs" / "gone.s1"
    agent.mkdir(parents=True)
    (agent / "BASE.json").write_text(json.dumps({"checkpoint": "local-files/cp_999", "values": [1.0, 0.5]}))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "value_axis_sweep.py"),
            "--data",
            str(tmp_path),
            "--agent",
            "gone.s1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode != 0
    assert "which is not there" in result.stderr


def test_sweeping_one_objective_is_allowed_but_halves_the_arms(volume: Path) -> None:
    out = dry_run(volume, "novalue11.s1234", "--objectives", "1")

    assert "objectives   [1]" in out
    assert "arms         25 x 750,000 steps" in out


def test_the_estimate_scales_with_arm_length(volume: Path) -> None:
    """Asserted as a ratio, not a literal: the hours depend on MEASURED_SPS, which
    is a property of the hardware and was retuned once already."""
    import re

    def hours(out: str) -> float:
        return float(re.search(r"~([\d.]+) h", out).group(1))

    for steps in (400_000, 800_000):
        expected = 50 * steps / sweep.MEASURED_SPS / 3600
        assert hours(dry_run(volume, "novalue11.s1234", "--steps", str(steps))) == pytest.approx(expected, abs=0.05)


def test_offsets_can_be_given_to_reproduce_an_earlier_grid(volume: Path) -> None:
    """A replication has to use the grid it is replicating, not the current default."""
    out = dry_run(volume, "novalue11.s1234", "--objectives", "1", "--offsets", "0.2", "0.4")

    assert "o1+040@750k" in out
    assert "o1-020@750k" in out
    assert "o1+045@750k" not in out
    assert "arms         5 x 750,000 steps" in out


def test_reordering_can_be_allowed_explicitly(volume: Path) -> None:
    """Refused by default; deliberate for three objectives, where the grid must
    span rank changes because a one-difference task cannot need two dimensions."""
    # Objective 0 down by 0.55 lands at 0.45, below objective 1's 0.5, so the
    # ranking changes. Swept the other way it would mirror to a negative reward,
    # which values_tag refuses outright.
    out = dry_run(volume, "novalue11.s1234", "--objectives", "0", "--offsets", "0.55", "--allow-reorder")

    assert "reorder the objectives, allowed explicitly" in out
    assert "o0-055@750k" in out


def test_an_arm_that_stopped_early_is_retrained_not_counted(volume: Path) -> None:
    """A killed arm leaves the checkpoints it had. Skipping it would fit a 200k
    arm as though it had run to 400k, under a name claiming 400k."""
    arm = volume / "runs" / "novalue11.s1234" / "arms" / "o1+045@400k" / "local-files"
    arm.mkdir(parents=True)
    (arm / "cp_046080").mkdir()
    (arm / "cp_199680").mkdir()

    out = dry_run(volume, "novalue11.s1234", "--steps", "400000")

    assert "o1+045@400k stopped at 199,680 of 400,000, retraining" in out
    assert "o1+045@400k already complete" not in out


def test_an_arm_that_reached_its_length_is_left_alone(volume: Path) -> None:
    arm = volume / "runs" / "novalue11.s1234" / "arms" / "o1+045@400k" / "local-files"
    arm.mkdir(parents=True)
    (arm / "cp_399360").mkdir()

    assert "o1+045@400k already complete" in dry_run(volume, "novalue11.s1234", "--steps", "400000")
