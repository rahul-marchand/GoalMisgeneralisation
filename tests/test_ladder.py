"""Rungs are named for where in training they stand, and resolved by number."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goalmisgen.ladder import base_payload, make_rung, plan_rung
from goalmisgen.volume import parse_checkpoint_dirname, rung_agent_name

CONFIG = {
    "cfg": {
        "train_env": {"objective_values": [1.0, 0.5], "n_objectives": 2},
        "total_timesteps": 150_000_000,
    }
}


def write_run(volume: Path, agent: str, checkpoints: list[str]) -> Path:
    directory = volume / "runs" / agent / "local-files"
    for name in checkpoints:
        (directory / name).mkdir(parents=True)
        (directory / name / "cfg.json").write_text(json.dumps(CONFIG))
    return volume / "runs" / agent


def test_a_checkpoint_name_reads_back_as_a_number_whatever_its_padding() -> None:
    # The same 70.1M steps, as a 150M run and an 80M run each write it.
    assert parse_checkpoint_dirname("cp_070103040") == 70_103_040
    assert parse_checkpoint_dirname("cp_70103040") == 70_103_040
    assert parse_checkpoint_dirname("local-files") is None
    assert parse_checkpoint_dirname("cp_") is None


def test_a_rung_is_named_for_the_step_not_the_directory() -> None:
    """Both paddings give one name, so one point in training is one rung."""
    assert rung_agent_name("novalue11.s1234", "cp_070103040") == "novalue11.s1234.at70103040"
    assert rung_agent_name("threeobj.even.s1234", "cp_70103040") == "threeobj.even.s1234.at70103040"


def test_checkpoints_past_100M_get_a_real_name() -> None:
    """The shell version stripped a literal ``cp_0``, which these have not got.

    ``novalue11.s1234.atcp_100147200`` is what that produced, and it would have
    been created rather than refused — a rung whose name carries the prefix it
    was meant to lose, sorting nowhere near its own ladder.
    """
    assert rung_agent_name("novalue11.s1234", "cp_100147200") == "novalue11.s1234.at100147200"


def test_a_rung_records_its_own_position_not_the_run_s_total() -> None:
    """``steps`` is the number a reader of a ladder wants, and it is the rung's."""
    rung = plan_rung("novalue11.s1234", "cp_020029440")
    payload = base_payload(rung, CONFIG)
    assert payload["steps"] == 20_029_440
    assert payload["source_total_timesteps"] == 150_000_000
    assert payload["checkpoint"] == "local-files/cp_020029440"
    assert payload["values"] == [1.0, 0.5]


def test_make_rung_points_at_the_checkpoints_it_shares(tmp_path: Path) -> None:
    write_run(tmp_path, "novalue11.s1234", ["cp_020029440", "cp_140206080"])
    rung = make_rung(tmp_path, "novalue11.s1234", "cp_020029440")

    directory = tmp_path / "runs" / rung.agent
    assert (directory / "local-files").is_symlink()
    # Relative, so the ladder survives the volume being mounted somewhere else.
    assert not Path((directory / "local-files").readlink()).is_absolute()
    assert (directory / "local-files" / "cp_020029440").is_dir()
    assert json.loads((directory / "BASE.json").read_text())["steps"] == 20_029_440


def test_make_rung_is_idempotent(tmp_path: Path) -> None:
    """A ladder interrupted half way is re-run, and must not disturb its arms."""
    write_run(tmp_path, "novalue11.s1234", ["cp_020029440"])
    rung = make_rung(tmp_path, "novalue11.s1234", "cp_020029440")
    marker = tmp_path / "runs" / rung.agent / "BASE.json"
    marker.write_text(json.dumps({"checkpoint": "local-files/cp_020029440", "values": [1.0, 0.5], "edited": True}))

    make_rung(tmp_path, "novalue11.s1234", "cp_020029440")
    assert json.loads(marker.read_text())["edited"] is True


def test_a_checkpoint_that_is_not_there_says_what_is(tmp_path: Path) -> None:
    """The failure the hardcoded names produced, made loud instead of silent."""
    write_run(tmp_path, "novalue11.s1234", ["cp_100147200"])
    with pytest.raises(FileNotFoundError, match="cp_100147200"):
        make_rung(tmp_path, "novalue11.s1234", "cp_100146560")


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    write_run(tmp_path, "novalue11.s1234", ["cp_020029440"])
    rung = make_rung(tmp_path, "novalue11.s1234", "cp_020029440", dry_run=True)
    assert not (tmp_path / "runs" / rung.agent).exists()
