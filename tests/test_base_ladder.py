"""Tests for the ladder driver's planning.

Same principle as ``tests/test_value_axis_sweep.py``: the driver spends GPU-hours
per invocation, so what is worth testing is everything it decides *before* it
commits — which checkpoint each rung landed on, and what that rung would cost.

``--dry-run`` is the surface under test, and it has to work without writing
anything, because checking a ladder before paying for it is the whole point of
having the flag.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The real save schedule of a 150M run: every ~1M to 20M, then every ~10M. The
# gaps are what makes --near necessary, so the fixture has to have them.
CHECKPOINTS = [
    "cp_002001920",
    "cp_005007360",
    "cp_010014720",
    "cp_020029440",
    "cp_030044160",
    "cp_100147200",
    "cp_140206080",
]

CONFIG = {
    "cfg": {
        "train_env": {"objective_values": [1.0, 0.5], "n_objectives": 2},
        "total_timesteps": 150_000_000,
    }
}


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    for name in CHECKPOINTS:
        directory = tmp_path / "runs" / "novalue11.s1234" / "local-files" / name
        directory.mkdir(parents=True)
        (directory / "cfg.json").write_text(json.dumps(CONFIG))
    return tmp_path


def ladder(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "base_ladder.py"), "--data", str(root), "--agent", "novalue11.s1234", *extra],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_near_resolves_to_the_checkpoint_that_exists(volume: Path) -> None:
    """Asked for 20M, gets the 20,029,440 the run actually saved."""
    result = ladder(volume, "--near", "20", "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cp_020029440" in result.stdout
    assert "novalue11.s1234.at20029440" in result.stdout


def test_the_100M_rung_resolves_rather_than_being_skipped(volume: Path) -> None:
    """campaign.sh asked for cp_100146560, which is not a checkpoint of anything.

    It tested for the directory and printed "not saved, skipping that rung", so
    the stage reported success having done half of what it claims.
    """
    result = ladder(volume, "--near", "100", "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cp_100147200" in result.stdout
    assert "novalue11.s1234.at100147200" in result.stdout


def test_how_far_a_rung_landed_from_what_was_asked_is_reported(volume: Path) -> None:
    """A rung is only as good as its distance from the point it stands for."""
    result = ladder(volume, "--near", "27", "--dry-run")

    # 27M falls between saves at 20.0M and 30.0M; nearest wins and says so.
    assert "cp_030044160" in result.stdout
    assert "off by" in result.stdout


def test_two_requests_landing_on_one_checkpoint_make_one_rung(volume: Path) -> None:
    """Otherwise the same axis appears twice on the plot and reads as a replication."""
    result = ladder(volume, "--near", "21", "22", "--dry-run")

    assert "keeping one rung" in result.stdout
    assert "\n1 rung:" in result.stdout


def test_a_dry_run_writes_nothing(volume: Path) -> None:
    result = ladder(volume, "--near", "20", "40", "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (volume / "runs" / "novalue11.s1234.at20029440").exists()
    assert "Nothing was written" in result.stdout


def test_a_dry_run_counts_the_arms_and_the_hours(volume: Path) -> None:
    """Both objectives is 50 arms a rung, which is the number that costs money."""
    result = ladder(volume, "--near", "20", "--dry-run")

    assert "50" in result.stdout
    assert "arms to train" in result.stdout


def test_arms_already_trained_are_not_counted_again(volume: Path) -> None:
    """A ladder split across GPUs is re-run constantly; it must show what is left."""
    arm = volume / "runs" / "novalue11.s1234.at20029440" / "arms" / "o1+045@400k" / "local-files" / "cp_400000"
    arm.mkdir(parents=True)

    result = ladder(volume, "--near", "20", "--objectives", "1", "--dry-run")

    # 25 arms in a one-objective sweep, one of them done.
    assert "24" in result.stdout


def test_a_ladder_with_no_rungs_asked_for_is_refused(volume: Path) -> None:
    result = ladder(volume, "--dry-run")

    assert result.returncode != 0
    assert "there is no default ladder" in result.stderr


def test_a_checkpoint_name_that_is_not_one_is_refused(volume: Path) -> None:
    result = ladder(volume, "--checkpoints", "cp_100146560", "--dry-run")

    assert result.returncode != 0
    assert "has no cp_100146560" in result.stderr or "cp_100147200" in result.stderr
