"""Recording what an agent is thinking, alongside what it then does.

The DRC's carry *is* its working state: three ConvLSTM layers, each holding a
32-channel vector for every maze cell. That spatial structure is what makes a
per-cell linear probe meaningful — the same readout is applied at every
position, so a probe that works is finding something the network represents
*about that cell*, not a global summary. A ResNet's feature maps and a
transformer's per-token residual stream have the same structure, and
:mod:`goalmisgen.nets.readers` hands all three over in the same shape, so
nothing here is specific to the DRC any more.

Activations are captured **before the agent moves**. A plan that only appears
once the agent is halfway there is not a plan; the claim worth testing is that
the route is already present at the first step.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import numpy as np

from goalmisgen.envs.observation import AGENT_CHANNEL, WALL_CHANNEL
from goalmisgen.envs.solver import distance_field
from goalmisgen.nets.readers import state_reader_for


@dataclasses.dataclass
class Rollout:
    """One episode: what the network held at t=0, and what the agent then did."""

    features: np.ndarray
    """(height, width, n_layers * channels) - per-cell hidden state at t=0."""

    cell_state: np.ndarray
    """(height, width, n_layers * channels) - per-cell ConvLSTM cell state at t=0.

    Captured alongside ``features`` rather than in a separate pass because the
    two come from one forward pass; collecting them apart would cost a second
    rollout and risk the two describing different episodes.

    A network that carries nothing between steps has no cell state; for those
    this holds the same array as ``features``, so code that reads shapes keeps
    working and code that writes into a carry has nothing to write to.
    """

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

    distance: np.ndarray
    """(height, width) int - BFS distance from the agent's start, UNREACHABLE for walls.

    A cell on the route at step ``k`` is ``k`` steps away, so without this the
    only available negatives are cells at *every* distance and any feature
    encoding "how far is this cell" scores well with no plan in it at all.
    """

    info: dict
    """Episode outcome and ground truth, from the environment's final info."""


def stack_layers(carry, index: int, state: str = "h") -> np.ndarray:
    """Concatenate one recurrent variable across every layer, for one environment.

    ``h`` is what the layer exposes to the next layer and to the head, so it is
    the state the rest of the network can read, and it is what a probe asking
    "is this information available" should look at.

    ``c`` is the cell state — the layer's persistent memory, carried across
    ticks and across environment steps rather than recomputed from the gates
    each time. It is the site the planning-interpretability interventions write
    to, and the distinction is causal rather than cosmetic: an edit to ``h`` is
    overwritten by the next tick, an edit to ``c`` is something the recurrence
    has to carry.

    Returns (height, width, n_layers * channels).
    """
    if state not in ("h", "c"):
        raise ValueError(f"state must be 'h' or 'c', got {state!r}")
    return np.concatenate([np.asarray(getattr(layer, state))[index] for layer in carry], axis=-1)


@dataclasses.dataclass(frozen=True)
class Capture:
    """One forward pass, fully named — what produced the state being probed.

    Three things are conflated if this is left implicit, and they cost very
    differently: whose parameters compute the probed state (a GPU collection),
    how many extra ticks run first (a GPU collection), and which grid is read
    off the result (free). Naming the first two lets rollouts be reused across
    every arm that shares them.

    ``actor`` and ``reader`` are separate because they answer different
    questions. With one agent acting and another's state probed, the labels are
    held fixed while the representation varies — which is what comparing
    checkpoints requires. Arms in one table must share an ``actor``, or their
    labels differ and the comparison is unattributable.
    """

    name: str
    reader: str
    """Key into the caller's parameter registry: whose state is probed."""

    actor: str = "agent"
    """Whose actions generate the episodes, and therefore the labels."""

    steps_to_think: int = 0


class RolloutCache:
    """Collects each distinct capture once and hands the same rollouts back.

    The pilot collected four sets per thinking value where two would do, and
    said so only in a comment. Keyed on the capture and the seed, so a table of
    six arms over two captures costs two collections.
    """

    def __init__(self, collect):
        self._collect = collect
        self._cache: dict[tuple, list[Rollout]] = {}

    def get(self, capture: Capture, seed: int, n_episodes: int) -> list[Rollout]:
        key = (capture, seed, n_episodes)
        if key not in self._cache:
            self._cache[key] = self._collect(capture, seed, n_episodes)
        return self._cache[key]

    @property
    def collections(self) -> int:
        """How many rollout sets were actually gathered. Asserted in tests."""
        return len(self._cache)


def require_one_actor(captures) -> None:
    """Arms compared in one table must share an actor.

    Different actors mean different episodes and therefore different labels, so
    a difference between arms could not be attributed to the representation.
    """
    actors = sorted({capture.actor for capture in captures})
    if len(actors) > 1:
        raise ValueError(
            f"arms in one table must share an actor, got {actors}; the labels would differ and any "
            "difference between arms would be unattributable"
        )


def collect_rollouts(
    envs,
    policy,
    params,
    n_episodes: int,
    seed: int = 0,
    steps_to_think: int = 0,
    probe_params=None,
    probe_steps_to_think: int = 0,
    steer_delta=None,
) -> list[Rollout]:
    """Run episodes, capturing the hidden state at t=0 and the route taken.

    ``steps_to_think`` is the number of *extra* passes over the initial
    observation before acting, on top of the one the agent always makes. Each
    pass runs the DRC's three internal ticks. A network with no state between
    steps computes the same thing on every pass, so for it thinking passes —
    the agent's and the probe's alike — change nothing, and steering has no
    carry to write into and is refused.

    The two ``probe_`` arguments change *what is probed* without changing what
    the agent does, which is what makes them controls:

    ``probe_params`` reads the features out of a different network - an
    untrained one of the same shape is the baseline that matters, since a random
    convolutional tower already beats a pointwise readout of the observation by
    virtue of having a receptive field at all.

    ``probe_steps_to_think`` adds thinking passes for the probe alone. Adding
    them to ``steps_to_think`` instead changes the route the agent walks, so the
    labels move with the features and any change in accuracy is unattributable.
    """
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    # The same call with the per-cell state of the pass returned alongside it.
    # Which grids those are depends on the architecture; the reader knows.
    reader = state_reader_for(policy)
    if steer_delta is not None and not reader.has_cell_state:
        raise ValueError("steering writes into the recurrent cell state, and this network has none")
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
        carry, first_action, _, key, state = reader.step(params, carry, observations, starts, key, temperature=0.0)

        # Everything below feeds the probe only. `key` is deliberately not
        # advanced and `carry` is not reassigned, so the actions the agent goes
        # on to take - and therefore the labels - are identical whatever is
        # probed.
        features_params = params if probe_params is None else probe_params
        if probe_params is None:
            probe_carry = carry
        else:
            probe_carry = policy.apply(probe_params, key, envs.observation_space.shape, method=policy.initialize_carry)
            probe_carry, _, _, _, state = reader.step(features_params, probe_carry, observations, starts, key, temperature=0.0)
        # Applied *before* the extra passes, so probe_steps_to_think measures
        # how much of the displacement survives them. Steering the arithmetic is
        # easy; showing the shift outlives nine gated recurrent updates is what
        # decides whether a behavioural null means "ignored" or "did not stick".
        if steer_delta is not None:
            from goalmisgen.analysis.steering import apply_to_carry

            probe_carry = apply_to_carry(probe_carry, steer_delta)
            state = reader.state_of_carry(probe_carry)

        for _ in range(probe_steps_to_think):
            probe_carry, _, _, _, state = reader.step(features_params, probe_carry, observations, starts, key, temperature=0.0)

        initial_obs = np.asarray(observations)

        # NCHW from the wrapper; probes want the spatial axes last.
        height, width = initial_obs.shape[2], initial_obs.shape[3]
        visited = np.zeros((envs.num_envs, height, width), dtype=bool)
        visit_step = np.full((envs.num_envs, height, width), -1, dtype=np.int16)
        finals: list[dict | None] = [None] * envs.num_envs
        done = np.zeros(envs.num_envs, dtype=bool)

        agent_channel = AGENT_CHANNEL
        here = initial_obs[:, agent_channel] > 0.5
        visited |= here
        visit_step[here] = 0

        # Bounded by the environment's own limit rather than a magic number:
        # if the loop ever exits early, `finals` stays None and the outer
        # while-loop retries the same seed forever.
        step_budget = getattr(envs, "max_episode_steps", None) or 512
        action = first_action
        for step_index in range(step_budget + 1):
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
        else:
            raise RuntimeError(
                f"episodes did not finish within {step_budget} steps; the step limit could not be read "
                "from the environment, so the capture loop would have retried this seed forever"
            )

        for index in range(envs.num_envs):
            final = finals[index]
            if final is None or len(rollouts) >= n_episodes:
                continue
            features, cell_state = state.stacked(index)
            rollouts.append(
                Rollout(
                    features=features,
                    cell_state=cell_state,
                    observation=np.moveaxis(initial_obs[index], 0, -1),
                    visited=visited[index],
                    visit_step=visit_step[index],
                    distance=_distance_from_start(initial_obs[index]),
                    info=final,
                )
            )

    return rollouts[:n_episodes]


def _distance_from_start(observation: np.ndarray) -> np.ndarray:
    """BFS distance from the agent to every free cell, read off the observation.

    Taken from the observation rather than the level so the capture path keeps
    no extra state; the wall and agent channels are all a shortest-path search
    needs.
    """
    walls = observation[WALL_CHANNEL] > 0.5
    start = np.argwhere(observation[AGENT_CHANNEL] > 0.5)[0]
    return distance_field(walls, (int(start[0]), int(start[1])))
