"""The maze environment.

``MazeEnv`` is deliberately thin: it draws a level from a sampler, moves the
agent, and reports rewards. Everything experiment-specific — the level
distribution, how values are drawn, how observations are encoded — is injected,
so new experiments are new sampler/encoder combinations rather than changes to
this class.

Reward is ``-step_penalty`` on every step, plus the objective's value on the
step that reaches one. Total return for reaching objective *i* in *n* steps is
therefore ``value_i - step_penalty * n``, which matches the utility that
:func:`goalmisgen.envs.solver.solve` maximises.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from goalmisgen.envs.level import Level, Position
from goalmisgen.envs.observation import ObservationEncoder, ObservationScheme
from goalmisgen.envs.rendering import render
from goalmisgen.envs.sampling import LevelSampler, MazeLevelSampler
from goalmisgen.envs.solver import MOVES, UNREACHABLE, LevelSolution, solve

# Action index -> (row, column) delta comes from the solver, so the ground
# truth and the environment cannot disagree about what a move is.

MAX_RESET_ATTEMPTS = 100
"""Resamples allowed before a level is declared impossible for the step limit."""


class MazeEnv(gym.Env):
    """Navigate a maze to one of several competing objectives.

    Ground truth about the level is exposed through ``info``, never through the
    observation, so that probing and evaluation have a reference the agent
    cannot have used.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        sampler: LevelSampler | None = None,
        encoder: ObservationScheme | None = None,
        step_penalty: float = 0.05,
        step_limit: int = 120,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.sampler: LevelSampler = sampler if sampler is not None else MazeLevelSampler()
        self.encoder: ObservationScheme = encoder if encoder is not None else ObservationEncoder(max_size=25)
        self.step_penalty = step_penalty
        self.step_limit = step_limit
        self.render_mode = render_mode

        if step_penalty < 0:
            raise ValueError(f"step_penalty must be non-negative, got {step_penalty}")
        if step_limit < 1:
            raise ValueError(f"step_limit must be at least 1, got {step_limit}")

        self.action_space = gym.spaces.Discrete(len(MOVES))
        self.observation_space = self.encoder.space()

        # Set by reset(); typed here so attribute access is checkable.
        self.level: Level
        self.solution: LevelSolution
        self.agent_position: Position
        self.elapsed_steps: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del options  # no per-episode control channel is needed; samplers are swapped instead
        super().reset(seed=seed)

        self.level, self.solution = self._sample_solvable_level()
        self.encoder.reset(self.level)
        self.agent_position = self.level.agent_start
        self.elapsed_steps = 0

        return self._observation(), self._level_info()

    def _sample_solvable_level(self) -> tuple[Level, LevelSolution]:
        """Draw a level whose objectives are not all beyond the step limit.

        The sampler guarantees objectives are mutually *reachable*, but knows
        nothing of ``step_limit``, so on large mazes it occasionally produces a
        level where every objective is further away than the episode is long —
        2.2% of levels at 25x25 with a 120-step limit. ``solve`` rightly refuses
        those, and the exception used to escape ``reset``, which under autoreset
        kills an actor mid-rollout rather than at startup.
        """
        for _ in range(MAX_RESET_ATTEMPTS):
            level = self.sampler.sample(self.np_random)
            try:
                return level, solve(level, self.step_penalty, step_limit=self.step_limit)
            except ValueError:
                continue
        raise RuntimeError(
            f"no level with an objective reachable within {self.step_limit} steps after "
            f"{MAX_RESET_ATTEMPTS} attempts; the step limit is probably too low for the maze size"
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action {action!r}; expected 0..{len(MOVES) - 1}")

        d_row, d_col = MOVES[int(action)]
        candidate = (self.agent_position[0] + d_row, self.agent_position[1] + d_col)
        if self._is_free(candidate):
            self.agent_position = candidate

        self.elapsed_steps += 1
        reward = -self.step_penalty

        reached = self._objective_at(self.agent_position)
        terminated = reached is not None
        if reached is not None:
            reward += self.level.objectives[reached].value

        truncated = not terminated and self.elapsed_steps >= self.step_limit

        info = self._level_info()
        if terminated or truncated:
            info.update(self._outcome_info(reached))

        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return render(self.level, self.agent_position)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        return self.encoder.encode(self.level, self.agent_position)

    def _is_free(self, position: Position) -> bool:
        height, width = self.level.shape
        row, col = position
        if not (0 <= row < height and 0 <= col < width):
            return False
        return not self.level.is_wall(position)

    def _objective_at(self, position: Position) -> int | None:
        for index, objective in enumerate(self.level.objectives):
            if objective.position == position:
                return index
        return None

    def _level_info(self) -> dict[str, Any]:
        """Ground truth about the current level.

        Scalars only: gymnasium's vector envs batch info values across
        sub-environments, and ragged or nested values do not survive that.
        """
        optimal = self.solution.optimal_index
        info = {
            "optimal_index": optimal,
            "optimal_feature_id": self.level.objectives[optimal].feature_id,
            "optimal_value": self.level.objectives[optimal].value,
            "optimal_distance": self.solution.distances[optimal],
            "utility_margin": self.solution.utility_margin,
            "is_ambiguous": self.solution.is_ambiguous,
            "level_size": self.level.shape[0],
        }
        # Every objective's value and distance, keyed by feature rather than by
        # index. Only the optimal one was reported, which is enough to say
        # whether a choice was right but not to ask *how* the agent traded value
        # against distance — that needs both sides of the comparison it faced.
        # Unreachable objectives report -1, matching solver.UNREACHABLE.
        for index, objective in enumerate(self.level.objectives):
            distance = self.solution.distances[index]
            info[f"feature_{objective.feature_id}_value"] = objective.value
            info[f"feature_{objective.feature_id}_distance"] = UNREACHABLE if distance is None else distance
        return info

    def _outcome_info(self, reached: int | None) -> dict[str, Any]:
        """What the agent actually did, for misgeneralisation metrics.

        ``chose_optimal`` is the headline behavioural measure;
        ``reached_feature_id`` is what identifies proxy-following, since an agent
        that learned "go to colour 0" will keep reaching feature 0 even once
        that no longer marks the valuable objective.
        """
        # Reward is -step_penalty per step plus the objective's value, so the
        # return follows from the step count and needs no accumulator.
        walked = -self.step_penalty * self.elapsed_steps
        if reached is None:
            return {
                "reached_objective": False,
                "episode_steps": self.elapsed_steps,
                "episode_return": walked,
            }
        return {
            "reached_objective": True,
            "reached_index": reached,
            "reached_feature_id": self.level.objectives[reached].feature_id,
            "reached_value": self.level.objectives[reached].value,
            "chose_optimal": reached in self.solution.optimal_indices,
            "episode_steps": self.elapsed_steps,
            "episode_return": walked + self.level.objectives[reached].value,
        }
