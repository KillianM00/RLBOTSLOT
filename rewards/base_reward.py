"""
BaseReward: abstract base class for all reward components.
Mirrors rlgym RewardFunction interface and adds:
  - weight          (float) — scaling factor used by DynamicCombinedReward
  - active_from_stage (int)  — curriculum stage when this reward activates
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg


class BaseReward:
    """
    Minimal reward function base compatible with rlgym RewardFunction.
    When rlgym is installed, individual rewards inherit from this class
    and call super().reset() / super().get_rewards() as needed.
    """

    weight: float = 1.0
    active_from_stage: int = 0

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        pass

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        return {agent: 0.0 for agent in agents}

    # ------------------------------------------------------------------
    # Convenience helpers shared across subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _car_pos(state: Any, agent_id: Any) -> np.ndarray:
        try:
            return np.asarray(state.cars[agent_id].physics.position, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _car_vel(state: Any, agent_id: Any) -> np.ndarray:
        try:
            return np.asarray(state.cars[agent_id].physics.linear_velocity, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _ball_pos(state: Any) -> np.ndarray:
        try:
            return np.asarray(state.ball.position, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _ball_vel(state: Any) -> np.ndarray:
        try:
            return np.asarray(state.ball.linear_velocity, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _team(state: Any, agent_id: Any) -> int:
        try:
            return int(state.cars[agent_id].team_num)
        except Exception:
            return 0
