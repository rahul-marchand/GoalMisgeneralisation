"""Tests for reading a container's real CPU allowance.

Getting this wrong is silent and expensive: the fallback is the host's core
count, so a pod with a five-core quota happily starts forty-eight workers, each
gets a tenth of a core, and generation simply takes several times longer with
nothing in the output to say why.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_levels import usable_cpus  # noqa: E402


def write(root: Path, relative: str, text: str) -> None:
    path = root / "sys/fs/cgroup" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_reads_a_cgroup_v2_quota(tmp_path):
    write(tmp_path, "cpu.max", "510000 100000\n")
    assert usable_cpus(tmp_path) == 5


def test_reads_a_cgroup_v1_quota(tmp_path):
    """The case that was missed, and that cost a run its worker count."""
    write(tmp_path, "cpu/cpu.cfs_quota_us", "510000\n")
    write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    assert usable_cpus(tmp_path) == 5


def test_v2_saying_unlimited_falls_back_to_the_core_count(tmp_path):
    write(tmp_path, "cpu.max", "max 100000\n")
    assert usable_cpus(tmp_path) == mp.cpu_count()


def test_v1_saying_unlimited_falls_back_to_the_core_count(tmp_path):
    """An unlimited v1 quota is -1, which must not become a worker count."""
    write(tmp_path, "cpu/cpu.cfs_quota_us", "-1\n")
    write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    assert usable_cpus(tmp_path) == mp.cpu_count()


def test_no_cgroup_at_all_falls_back_to_the_core_count(tmp_path):
    assert usable_cpus(tmp_path) == mp.cpu_count()


def test_a_quota_below_one_core_still_gives_a_worker(tmp_path):
    write(tmp_path, "cpu/cpu.cfs_quota_us", "50000\n")
    write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    assert usable_cpus(tmp_path) == 1
