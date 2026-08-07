"""Tests for the route-as-direction concept and the plan written back into it.

The property that matters is that a written plan is *walkable*: following the
directions from the agent's cell has to arrive at the objective. An edit that
does not describe a route is not an intervention on a plan, it is noise with a
spatial pattern — and it would produce exactly the same shaped null we already
have five of.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from goalmisgen.analysis import geometry, plans
from goalmisgen.envs.observation import ObservationEncoder
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import MOVES, path_to_objective


@dataclasses.dataclass
class FakeRollout:
    observation: np.ndarray
    visited: np.ndarray
    visit_step: np.ndarray


def sampled(n, size=11, seed=0):
    sampler = MazeLevelSampler(size_range=(size, size))
    encoder = ObservationEncoder(max_size=size, n_features=2)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        level = sampler.sample(rng)
        yield level, encoder.encode(level, level.agent_start)


def walk(labels: np.ndarray, start: tuple[int, int], limit: int = 400) -> tuple[int, int]:
    """Follow the written directions from ``start`` until they run out."""
    current = start
    for _ in range(limit):
        label = labels[current]
        if label < 0 or label == plans.NEVER:
            return current
        d_row, d_col = MOVES[int(label)]
        current = (current[0] + d_row, current[1] + d_col)
    raise AssertionError("the plan loops; following it never terminates")


def test_a_written_plan_walks_from_the_agent_to_the_objective():
    """The intervention's whole content. If this fails it is writing noise."""
    checked = 0
    for _, observation in sampled(60):
        start = geometry.agent_cell(observation)
        for feature in range(2):
            labels = plans.planned_directions(observation, feature, n_features=2)
            if labels is None:
                continue
            assert walk(labels, start) == geometry.objective_cell(observation, feature)
            checked += 1
    assert checked > 60, f"only {checked} reachable objectives; the test is not exercising much"


def test_the_written_route_is_the_shortest_one():
    """A plan longer than the solver's would be a different claim about the agent."""
    for level, observation in sampled(40):
        for index, objective in enumerate(level.objectives):
            reference = path_to_objective(level, index)
            labels = plans.planned_directions(observation, objective.feature_id, n_features=2)
            if reference is None or labels is None:
                continue
            written = int(((labels >= 0) & (labels < plans.NEVER)).sum())
            assert written == len(reference) - 1


def test_the_objective_cell_is_not_labelled_never():
    """The end of the route has no next move, and calling that NEVER would teach
    the probe that the goal is off-plan — the opposite of what is true."""
    for _, observation in sampled(30):
        for feature in range(2):
            labels = plans.planned_directions(observation, feature, n_features=2)
            if labels is None:
                continue
            assert labels[geometry.objective_cell(observation, feature)] == plans.UNSCOREABLE


def test_walls_are_never_scoreable():
    for _, observation in sampled(30):
        labels = plans.planned_directions(observation, 0, n_features=2)
        if labels is None:
            continue
        assert (labels[~geometry.free_cells(observation)] == plans.UNSCOREABLE).all()


def test_a_walled_off_objective_gives_no_plan():
    """Reaching either objective ends the episode, so a route through one never
    arrives at the other. Returning an all-NEVER grid instead of None would write
    the erase half of the intervention over the whole maze with nothing to
    replace it."""
    observation = np.zeros((5, 5, 5), dtype=np.float32)
    observation[:, :, 0] = 1.0
    observation[1, 1:4, 0] = 0.0  # a corridor: agent, then objective 0, then objective 1
    observation[1, 1, 1] = 1.0
    observation[1, 2, 2] = 1.0
    observation[1, 3, 3] = 1.0

    assert plans.planned_directions(observation, 0, n_features=2) is not None
    assert plans.planned_directions(observation, 1, n_features=2) is None


def test_the_sampler_leaves_every_objective_plannable():
    """The sampler rejects levels with an objective walled off behind another, so
    the experiment never has to drop an episode for want of a route. Asserted
    because the intervention silently skips levels where a plan is None, and a
    change to that rejection rule would shrink the sample without saying so."""
    for _, observation in sampled(200):
        for feature in range(2):
            assert plans.planned_directions(observation, feature, n_features=2) is not None


def test_observed_directions_follow_the_route_the_agent_walked():
    """Labels come from arrival *order*, not from arrival step plus one: walking
    into a wall advances the step counter without moving the agent, so the steps
    have gaps and the naive reconstruction silently drops those cells."""
    _, observation = next(iter(sampled(1)))
    height, width = observation.shape[:2]

    route = [(1, 1), (1, 2), (1, 3), (2, 3)]
    visit_step = np.full((height, width), -1, dtype=np.int64)
    # Deliberate gap at 2: the agent bumped into a wall on that step.
    for cell, step in zip(route, [0, 1, 3, 4]):
        visit_step[cell] = step
    visited = visit_step >= 0

    labels = plans.observed_directions(FakeRollout(observation, visited, visit_step))
    assert labels[1, 1] == MOVES.index((0, 1))
    assert labels[1, 2] == MOVES.index((0, 1))
    assert labels[1, 3] == MOVES.index((1, 0))
    assert labels[2, 3] == plans.UNSCOREABLE


def test_off_route_free_cells_are_never():
    _, observation = next(iter(sampled(1)))
    height, width = observation.shape[:2]
    visit_step = np.full((height, width), -1, dtype=np.int64)
    visit_step[1, 1] = 0
    visit_step[1, 2] = 1
    labels = plans.observed_directions(FakeRollout(observation, visit_step >= 0, visit_step))

    free = geometry.free_cells(observation)
    off_route = free & (visit_step < 0)
    assert (labels[off_route] == plans.NEVER).all()
    assert (labels[~free] == plans.UNSCOREABLE).all()


def test_class_names_line_up_with_the_moves():
    """The direction classes ARE the action indices. If these ever drift, every
    written plan points somewhere other than where it says."""
    assert plans.CLASS_NAMES[: len(MOVES)] == ("up", "down", "left", "right")
    assert MOVES == ((-1, 0), (1, 0), (0, -1), (0, 1))
    assert plans.NEVER == len(MOVES)
