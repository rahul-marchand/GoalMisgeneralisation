"""Measuring agent behaviour against ground truth, and probing internals."""

from goalmisgen.analysis.activations import Rollout, collect_rollouts, stack_layers
from goalmisgen.analysis.behaviour import BehaviourSummary, bin_by_margin, collect_episode_outcomes, summarise
from goalmisgen.analysis.probes import (
    DistanceBand,
    ProbeResult,
    auc_interval,
    cell_dataset,
    probe,
    probe_by_distance,
)

__all__ = [
    "BehaviourSummary",
    "DistanceBand",
    "ProbeResult",
    "auc_interval",
    "Rollout",
    "bin_by_margin",
    "cell_dataset",
    "collect_episode_outcomes",
    "collect_rollouts",
    "probe",
    "probe_by_distance",
    "stack_layers",
    "summarise",
]
