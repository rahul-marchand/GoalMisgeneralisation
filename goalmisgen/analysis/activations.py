"""Recording what a DRC agent is thinking, alongside what it then does.

The DRC's carry *is* its working state: three ConvLSTM layers, each holding a
32-channel vector for every maze cell. That spatial structure is what makes a
per-cell linear probe meaningful — the same readout is applied at every
position, so a probe that works is finding something the network represents
*about that cell*, not a global summary.

Activations are captured **before the agent moves**. A plan that only appears
once the agent is halfway there is not a plan; the claim worth testing is that
the route is already present at the first step.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import numpy as np


@dataclasses.dataclass
class Rollout:
    """One episode: what the network held at t=0, and what the agent then did."""

    features: np.ndarray
    """(height, width, n_features) - per-cell activation vector at t=0."""

    observation: np.ndarray
    """(height, width, channels) - the same moment's input, for baseline probes."""

    visited: np.ndarray
    """(height, width) bool - cells the agent actually stepped on."""

    visit_step: np.ndarray
    """(height, width) int - step at which each cell was first reached, -1 if never.

    Distinguishes the cell the agent is about to move onto from the far end of
    its route. Both are positives for the plan probe, but only the second is
    evidence of a plan rather than of an imminent action.
    """

    info: dict
    """Episode outcome and ground truth, from the environment's final info."""


def stack_layers(carry, index: int) -> np.ndarray:
    """Concatenate the hidden state of every layer for one environment.

    Uses ``h`` rather than ``c``: ``h`` is what the layer exposes to the next
    layer and to the head, so it is the state the rest of the network can
    actually read. Returns (height, width, n_layers * channels).
    """
    hs = [np.asarray(layer.h)[index] for layer in carry]
    return np.concatenate(hs, axis=-1)


def collect_rollouts(
    envs,
    policy,
    params,
    n_episodes: int,
    seed: int = 0,
    steps_to_think: int = 0,
) -> list[Rollout]:
    """Run episodes, capturing the hidden state at t=0 and the route taken.

    ``steps_to_think`` is the number of *extra* passes over the initial
    observation before acting, on top of the one the agent always makes. Each
    pass runs the DRC's three internal ticks, so ``steps_to_think=0`` probes a
    state that has seen the maze once. If plans refine iteratively, accuracy
    should climb with this.
    """
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    key = jax.random.PRNGKey(seed)

    rollouts: list[Rollout] = []
    while len(rollouts) < n_episodes:
        observations, _ = envs.reset(seed=seed + len(rollouts))
        carry = policy.apply(params, key, envs.observation_space.shape, method=policy.initialize_carry)
        starts = np.zeros(envs.num_envs, dtype=bool)

        for _ in range(steps_to_think):
            carry, _, _, key = get_action(params, carry, observations, starts, key, temperature=0.0)

        # Probe *after* the network has processed the observation at least once.
        # initialize_carry returns zeros, so capturing before this would probe an
        # empty state - which shows up as an AUC of exactly 0.500.
        carry, first_action, _, key = get_action(params, carry, observations, starts, key, temperature=0.0)
        initial_carry = jax.tree_util.tree_map(np.asarray, carry)
        initial_obs = np.asarray(observations)

        # NCHW from the wrapper; probes want the spatial axes last.
        height, width = initial_obs.shape[2], initial_obs.shape[3]
        visited = np.zeros((envs.num_envs, height, width), dtype=bool)
        visit_step = np.full((envs.num_envs, height, width), -1, dtype=np.int16)
        finals: list[dict | None] = [None] * envs.num_envs
        done = np.zeros(envs.num_envs, dtype=bool)

        agent_channel = 1
        here = initial_obs[:, agent_channel] > 0.5
        visited |= here
        visit_step[here] = 0

        action = first_action
        for step_index in range(512):
            if step_index > 0:
                carry, action, _, key = get_action(params, carry, observations, starts, key, temperature=0.0)
            observations, _, terminated, truncated, info = envs.step(np.asarray(action))
            just_done = np.logical_or(terminated, truncated) & ~done

            # On the terminating step gymnasium has already autoreset, so
            # `observations` holds the *next* level. Reading it would record
            # that level's start cell and lose this episode's final cell - the
            # objective, which is the deepest point of the route the probe is
            # meant to find. `final_observation` carries the real last frame.
            # `final_observation` is left in HWC by cleanba's NHWC->NCHW
            # wrapper, which transposes the batched observation only.
            frames = np.asarray(observations).copy()
            finished = info.get("final_observation")
            for index in np.flatnonzero(just_done):
                if finished is not None and finished[index] is not None:
                    frames[index] = np.moveaxis(np.asarray(finished[index]), -1, 0)

            live = ~done
            here = (frames[:, agent_channel] > 0.5) & live[:, None, None]
            fresh = here & (visit_step < 0)
            visited |= here
            visit_step[fresh] = step_index + 1

            if just_done.any():
                # `final_info` is a numpy object array, so it must be tested
                # against None explicitly - `x or y` raises on an array.
                reported = info.get("final_info")
                for index in np.flatnonzero(just_done):
                    entry = reported[index] if reported is not None else None
                    finals[index] = dict(entry) if entry is not None else {}
                done |= just_done
            if done.all():
                break

        for index in range(envs.num_envs):
            final = finals[index]
            if final is None or len(rollouts) >= n_episodes:
                continue
            rollouts.append(
                Rollout(
                    features=stack_layers(initial_carry, index),
                    observation=np.moveaxis(initial_obs[index], 0, -1),
                    visited=visited[index],
                    visit_step=visit_step[index],
                    info=final,
                )
            )

    return rollouts[:n_episodes]
