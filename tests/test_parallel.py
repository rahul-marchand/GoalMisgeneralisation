"""Tests for the worker pools used by level and demonstration generation.

The property under test is the start method, not the arithmetic: forking a
process that has JAX loaded deadlocks, and a deadlock is invisible to a test
that only checks the answer. So the test asks the workers what they inherited.
Under fork they would see the parent's memory as it stood at the fork; under
spawn they start from a fresh interpreter and see only what was pickled to them.
"""

from __future__ import annotations

import multiprocessing as mp

from goalmisgen.parallel import worker_pool

INHERITED: str | None = None
"""Set by the parent *after* import, so only a forked child could see it."""


def read_inherited(_: int) -> str | None:
    return INHERITED


def double(x: int) -> int:
    return 2 * x


def test_workers_do_not_inherit_the_parent_process() -> None:
    global INHERITED
    INHERITED = "set in the parent"
    try:
        with worker_pool(2) as pool:
            assert pool.map(read_inherited, [0, 1]) == [None, None]
    finally:
        INHERITED = None


def test_the_pool_still_runs_work() -> None:
    with worker_pool(2) as pool:
        assert pool.starmap(double, [(1,), (2,), (3,)]) == [2, 4, 6]


def test_spawn_is_available_on_this_platform() -> None:
    """Guards the assumption the module rests on rather than the module itself."""
    assert "spawn" in mp.get_all_start_methods()
