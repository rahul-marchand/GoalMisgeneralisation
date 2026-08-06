"""What a probe is asked to predict, and what would answer it without computing.

A :class:`CellTarget` is one question, asked at every cell. It declares two
things, and the second is the point of the module:

``labels`` — what the probe should predict, with ``NaN`` at any cell that must
not be scored. Walls, unreachable cells and cells whose answer is trivially
visible in the input are all expressed this way, so the generic layer needs one
rule (drop non-finite rows) rather than a list of maze facts.

``confound`` — what a feature containing *no computation* would predict. This is
required, and it is what makes the controls structural. The target's labels are
the positive control's feature grid; its confound is the negative control's. So
:func:`controls` generates both from the target, and a new probe question cannot
be added to the codebase without declaring what would beat it for free.

That requirement exists because of a specific failure. The first distance-band
result was invalidated by a confound that had been raised in conversation and
waved through; a control that has to be remembered is a control that eventually
is not run.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Protocol, Sequence

import numpy as np

from goalmisgen.analysis import geometry
from goalmisgen.analysis.probes import Feature


class CellTarget(Protocol):
    """One probe question, asked at every cell, carrying its own null."""

    # Declared read-only so a frozen dataclass satisfies the protocol; a bare
    # attribute would require the implementation to be writable, which every
    # target here deliberately is not.
    @property
    def name(self) -> str:
        ...

    @property
    def confound_names(self) -> tuple[str, ...]:
        ...

    def labels(self, rollout) -> np.ndarray:
        """``(height, width)`` float. ``NaN`` marks a cell that must not be scored."""
        ...

    def confound(self, rollout) -> np.ndarray:
        """``(height, width, k)`` float, ``k >= 1``.

        Column 0 is the null's own prediction of ``labels``, in the same units,
        and must be a **lower bound** on them — the generic layer uses it to
        pick out the cells where the null is most wrong.
        """
        ...


Selector = Callable[[object], "int | None"]
"""Picks which objective an episode's field is measured to. ``None`` drops it."""


def fixed(feature_id: int) -> Selector:
    """Always the same objective, whatever the agent did."""

    def select(rollout) -> int | None:
        del rollout
        return feature_id

    return select


def reached(rollout) -> int | None:
    """The objective the agent actually went to. ``None`` if it timed out."""
    value = rollout.info.get("reached_feature_id")
    return None if value is None else int(value)


def unreached(rollout, n_features: int = 2) -> int | None:
    """The objective the agent ignored.

    Only defined for two objectives; with more there is no single "the other
    one" and the question needs restating rather than generalising.
    """
    if n_features != 2:
        raise ValueError(f"'unreached' is only defined for two objectives, got {n_features}")
    value = reached(rollout)
    return None if value is None else 1 - value


@dataclasses.dataclass(frozen=True)
class DistanceToObjective:
    """Shortest-path distance from each cell to one objective.

    Which objective is chosen per episode by ``select``, because *that keying is
    the experimental variable*: distance to a fixed feature, to the objective
    the agent reached, and to the one it ignored are the same measurement asked
    of three different quantities, and they come apart exactly where the
    interesting mechanism is.

    The null is free-space geometry — Manhattan and Chebyshev distance from the
    same source. Neither requires solving the maze, and on real levels Manhattan
    alone explains about a third of the variance in the true field, which is
    enough to look like a finding.
    """

    select: Selector
    name: str
    n_features: int = 2
    confound_names: tuple[str, ...] = ("manhattan", "chebyshev")

    def _source(self, rollout) -> tuple[int, int] | None:
        feature_id = self.select(rollout)
        return None if feature_id is None else geometry.objective_cell(rollout.observation, feature_id)

    def labels(self, rollout) -> np.ndarray:
        observation = rollout.observation
        geometry.check_layout(observation, self.n_features)
        feature_id = self.select(rollout)
        if feature_id is None:
            return np.full(observation.shape[:2], np.nan)

        source = geometry.objective_cell(observation, feature_id)
        field = geometry.bfs_field(geometry.blocking_walls(observation, feature_id, self.n_features), source)

        # Walls are already NaN. Drop the objective's own cell too: its distance
        # is zero and it is a one-hot channel in the input, so every arm scores
        # it correctly and none of them learn anything. Same trap as scoring the
        # route probe at step 0, where the observation baseline reached 1.000
        # because the agent's cell is literally an input channel.
        field[~geometry.free_cells(observation)] = np.nan
        field[field == 0] = np.nan
        return field

    def confound(self, rollout) -> np.ndarray:
        shape = rollout.observation.shape[:2]
        source = self._source(rollout)
        if source is None:
            return np.full((*shape, 2), np.nan)
        return np.stack([geometry.manhattan_field(shape, source), geometry.chebyshev_field(shape, source)], axis=-1)


def controls(target: CellTarget, rollouts: Sequence, seed: int = 0) -> tuple[Feature, ...]:
    """The arms that decide whether any other number in the table is readable.

    Generated from the target rather than written out beside it, so a probe
    question cannot be asked without them. In order: the field handed over, the
    no-computation null, and the field attached to the wrong maze.

    ``rollouts`` is needed only by the last one — permuting across episodes
    requires seeing all of them, which a per-rollout callable cannot.
    """
    return (
        Feature(f"oracle:{target.name}", lambda rollout: np.nan_to_num(target.labels(rollout))[:, :, None]),
        Feature(f"null:{'+'.join(target.confound_names)}", target.confound),
        _shuffled_oracle(target, rollouts, seed),
    )


def _shuffled_oracle(target: CellTarget, rollouts: Sequence, seed: int = 0) -> Feature:
    """The oracle field, attached to a different episode's maze.

    The positive control proves the rig can find a field it is handed, but it
    cannot catch a *consistent* misalignment between features and labels —
    the feature would be misaligned identically and still score 1.000. This
    catches it: the grids are real fields on the wrong mazes, so anything
    scoring above chance is reading something other than the maze in front of it.
    """
    order = np.random.default_rng(seed).permutation(len(rollouts))
    # A derangement, so no grid can land back on its own episode and pass.
    for position, destination in enumerate(order):
        if position == destination:
            swap = (position + 1) % len(order)
            order[position], order[swap] = order[swap], order[position]

    grids = {
        id(rollout): np.nan_to_num(target.labels(rollouts[order[index]]))[:, :, None] for index, rollout in enumerate(rollouts)
    }
    return Feature(f"shuffled:{target.name}", lambda rollout: grids[id(rollout)])


STEP_PENALTY = 0.05
"""What a step costs, so utility is ``value - STEP_PENALTY x distance``.

Matches the environment's default. If a run changes it, this must change with
it, or "the higher-utility objective" names a different objective than the one
the agent was trained to prefer.
"""


def objective_distance(rollout, feature_id: int, n_features: int = 2) -> float:
    """Steps from the agent to one objective, routing around the other.

    Routing around matters: reaching either objective ends the episode, so a
    path through the wrong one never arrives.
    """
    observation = rollout.observation
    field = geometry.bfs_field(
        geometry.blocking_walls(observation, feature_id, n_features),
        geometry.objective_cell(observation, feature_id),
    )
    return float(field[geometry.agent_cell(observation)])


def _pick(rollout, score, n_features: int) -> int | None:
    """The objective maximising ``score``. ``None`` on a tie or if one is cut off.

    Ties are dropped rather than broken. A tie means the two objectives are
    indistinguishable on this criterion, so an episode contributed under either
    label would be noise on both sides of a comparison.
    """
    values = []
    for feature in range(n_features):
        try:
            value = score(rollout, feature)
        except ValueError:
            return None
        if not np.isfinite(value):
            return None
        values.append(value)

    best = max(values)
    return None if values.count(best) > 1 else int(np.argmax(values))


def richer(rollout, n_features: int = 2) -> int | None:
    """The objective worth more, ignoring how far away it is."""
    return _pick(rollout, lambda r, f: geometry.objective_value(r.observation, f, n_features), n_features)


def nearer(rollout, n_features: int = 2) -> int | None:
    """The objective fewer steps away, ignoring what it is worth."""
    return _pick(rollout, lambda r, f: -objective_distance(r, f, n_features), n_features)


def best_utility(rollout, n_features: int = 2, step_penalty: float = STEP_PENALTY) -> int | None:
    """The objective an optimal agent would take: value minus the walk."""
    return _pick(
        rollout,
        lambda r, f: geometry.objective_value(r.observation, f, n_features) - step_penalty * objective_distance(r, f, n_features),
        n_features,
    )


def coinflip(rollout, seed: int = 0, n_features: int = 2) -> int:
    """A split with nothing behind it.

    The control for the comparison machinery itself: any difference between two
    sides assigned at random is a bug in how the sides are built, not a fact
    about the network. Deterministic per episode so a re-run is comparable.
    """
    return int(np.random.default_rng((seed, id(rollout) % 2**32)).integers(n_features))
