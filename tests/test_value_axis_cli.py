"""Tests for the fine-tune script's command line.

These exist because the three-objective sweep died on its first arm: --value was
still required while the three-objective path passes only --objective-values, so
every invocation failed in argparse before doing anything. The configuration
builder had been tested by handing it a Namespace directly, which is exactly the
path that cannot catch this — it skips the parser entirely.

So the rule these encode is that the *entry point* gets tested, not just the
function behind it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "013_value_axis.py"
spec = importlib.util.spec_from_file_location("value_axis", SCRIPT)
assert spec is not None and spec.loader is not None
value_axis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(value_axis)


def run(argv: list[str]):
    sys.argv = ["013_value_axis.py", *argv]
    return value_axis.parse_args()


BASE = ["cp", "--levels", "/tmp/levels", "--run-dir", "/tmp/run"]


def test_two_objectives_by_value_alone():
    args = run([*BASE, "--value", "0.7"])
    assert value_axis.finetune_config(args).train_env.objective_values == (1.0, 0.7)


def test_three_objectives_without_value():
    """The invocation the sweep actually makes, and that used to fail outright."""
    args = run([*BASE, "--objective-values", "1.0", "0.65", "0.3"])
    env = value_axis.finetune_config(args).train_env
    assert env.objective_values == (1.0, 0.65, 0.3)
    assert env.n_objectives == 3


def test_objective_values_wins_over_value():
    args = run([*BASE, "--value", "0.7", "--objective-values", "1.0", "0.5", "0.2"])
    assert value_axis.finetune_config(args).train_env.objective_values == (1.0, 0.5, 0.2)


def test_neither_is_refused():
    with pytest.raises(SystemExit):
        run(BASE)


def test_a_single_objective_value_is_refused():
    with pytest.raises(SystemExit):
        run([*BASE, "--objective-values", "1.0"])


def test_the_agent_never_sees_a_value_channel():
    """The whole design: colour is the only cue to what an objective is worth."""
    args = run([*BASE, "--objective-values", "1.0", "0.65", "0.3"])
    env = value_axis.finetune_config(args).train_env
    assert env.value_encoding == "none"
    assert env.colour_is_the_only_value_cue is True


def test_the_value_channel_is_read_from_the_base_not_assumed(tmp_path) -> None:
    """013 was written for novalue11 and hardcoded hide_values=True, so
    fine-tuning maze11 -- which has a value channel -- built a four-channel
    network against five-channel weights and died on a shape error."""
    import json

    novalue = tmp_path / "novalue"
    novalue.mkdir()
    (novalue / "cfg.json").write_text(json.dumps({"cfg": {"train_env": {"colour_is_the_only_value_cue": True}}}))
    withvalue = tmp_path / "withvalue"
    withvalue.mkdir()
    (withvalue / "cfg.json").write_text(json.dumps({"cfg": {"train_env": {"colour_is_the_only_value_cue": False}}}))

    assert value_axis.base_hides_values(novalue) is True
    assert value_axis.base_hides_values(withvalue) is False


def test_an_unreadable_base_config_keeps_the_old_behaviour(tmp_path) -> None:
    """Every existing sweep was on an agent without a value channel."""
    assert value_axis.base_hides_values(tmp_path) is True
