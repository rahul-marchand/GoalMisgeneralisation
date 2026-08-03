"""Tests for the Sokoban-assumption shims.

These guard a failure that is expensive to rediscover: training runs perfectly
for hundreds of thousands of steps, then dies at the first evaluation, and
because the exception is raised on the evaluation thread the process hangs
rather than exiting — so it reads as a stall, not a crash, while still billing
for the GPU.
"""

from __future__ import annotations

import cleanba.evaluate
import numpy as np

from goalmisgen.configs import compat


def test_shim_is_installed_on_import():
    assert getattr(cleanba.evaluate.get_cycles, compat._PATCH_FLAG, False)


def test_installing_twice_does_not_stack_wrappers():
    first = cleanba.evaluate.get_cycles
    compat.install()
    assert cleanba.evaluate.get_cycles is first


def test_symbolic_observations_yield_no_cycles():
    """Five channels is what our encoder produces; upstream asserts three."""
    observations = np.zeros((4, 5, 25, 25), dtype=np.float32)
    assert cleanba.evaluate.get_cycles(observations, last_box_time_step=3) == []


def test_upstream_assertion_would_have_fired():
    """Confirm the shim is not passing vacuously."""
    observations = np.zeros((4, 5, 25, 25), dtype=np.float32)
    try:
        compat._original_get_cycles(observations, last_box_time_step=3)
    except AssertionError:
        return
    raise AssertionError("expected upstream get_cycles to reject 5-channel observations")


def test_sokoban_shaped_observations_still_reach_upstream():
    """The shim must not silently disable the analysis for Sokoban itself."""
    calls = []
    original = compat._original_get_cycles
    try:
        compat._original_get_cycles = lambda obs, last_box_time_step: calls.append(obs.shape) or []
        rgb = np.zeros((6, 3, 10, 10), dtype=np.uint8)
        cleanba.evaluate.get_cycles(rgb, last_box_time_step=5)
    finally:
        compat._original_get_cycles = original
    assert calls == [(6, 3, 10, 10)]


def test_non_square_observations_are_skipped():
    observations = np.zeros((4, 3, 25, 31), dtype=np.float32)
    assert cleanba.evaluate.get_cycles(observations, last_box_time_step=3) == []
