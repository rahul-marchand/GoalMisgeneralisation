"""Measuring agent behaviour against ground truth, and probing internals."""

from goalmisgen.analysis.activations import Rollout, collect_rollouts, stack_layers
from goalmisgen.analysis.behaviour import BehaviourSummary, collect_episode_outcomes, summarise
from goalmisgen.analysis.probes import ProbeResult, cell_dataset, probe

__all__ = [
    "BehaviourSummary",
    "ProbeResult",
    "Rollout",
    "cell_dataset",
    "collect_episode_outcomes",
    "collect_rollouts",
    "probe",
    "stack_layers",
    "summarise",
]
