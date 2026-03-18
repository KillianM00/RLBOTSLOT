"""
GymWrapper: wraps rlgym 2.x RLGym (multi-agent dict interface) into the
single-process gym-compatible interface expected by rlgym-ppo.

rlgym-ppo expects:
  env.action_space        — gym.spaces.Discrete / MultiDiscrete / Box
  env.observation_space   — gym.spaces.Box
  env.reset()             — np.ndarray shape (n_agents, obs_dim) or (obs_dim,)
  env.step(action_array)  — (obs, rew_list, done_bool, truncated_bool, info)
  env.close()
  env.render()

rlgym 2.x provides dict-based multi-agent interface.
This wrapper bridges the two.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym              # type: ignore
    from gymnasium import spaces         # type: ignore

import config as cfg


class GymWrapper:
    """
    Wraps an rlgym 2.x RLGym environment into a gym-compatible interface.

    Multi-agent: reset() returns (n_agents, obs_dim) array;
    step() returns (obs_array, rew_list, done, truncated, info).
    """

    def __init__(self, rlgym_env):
        self._env = rlgym_env
        self._agent_ids: List[str] = []

        # Build spaces based on config
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(cfg.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(cfg.N_ACTIONS)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        obs_dict = self._env.reset()
        self._agent_ids = list(obs_dict.keys())
        obs_array = np.stack(
            [obs_dict[a].astype(np.float32) for a in self._agent_ids], axis=0
        )
        return obs_array  # (n_agents, obs_dim)

    # ------------------------------------------------------------------
    def step(
        self, action_array: np.ndarray
    ) -> Tuple[np.ndarray, List[float], bool, bool, Dict]:
        """
        action_array : (n_agents, action_size) — rlgym-ppo passes float array;
                       we cast to int for discrete lookup table.
        """
        # Build action dict: {agent_id: int_action}
        action_dict = {}
        action_array = np.asarray(action_array).reshape(-1)  # always (n_agents,)
        for i, agent_id in enumerate(self._agent_ids):
            action_dict[agent_id] = np.array([int(action_array[i])], dtype=np.int32)

        result = self._env.step(action_dict)

        # rlgym 2.x returns (obs_dict, rew_dict, terminated_dict, truncated_dict)
        obs_dict, rew_dict, terminated_dict, truncated_dict = result

        self._agent_ids = list(obs_dict.keys())
        obs_array = np.stack(
            [obs_dict[a].astype(np.float32) for a in self._agent_ids], axis=0
        )

        rew_list  = [float(rew_dict.get(a, 0.0)) for a in self._agent_ids]
        done      = any(terminated_dict.values())
        truncated = any(truncated_dict.values())

        return obs_array, rew_list, done, truncated, {}

    # ------------------------------------------------------------------
    def close(self):
        self._env.close()

    def render(self):
        if hasattr(self._env, "render"):
            self._env.render()

    def seed(self, seed: int = 0):
        pass
