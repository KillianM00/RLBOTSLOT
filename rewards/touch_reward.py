"""
TouchReward: reward for making contact with the ball.
Aerial touches are worth 3x standard ground touches.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import is_in_air


class TouchReward(BaseReward):
    """
    +1.0 per ground touch, +3.0 per aerial touch.
    Detects a touch by comparing ball.last_touch.player_index.
    """

    weight: float = 1.0
    active_from_stage: int = 0

    def __init__(self, aerial_multiplier: float = cfg.TOUCH_AERIAL_MULTIPLIER) -> None:
        self.aerial_multiplier = aerial_multiplier
        self._last_touch_tick: Dict[Any, int] = {}

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        self._last_touch_tick = {a: -1 for a in agents}

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        rewards: Dict[Any, float] = {}
        for agent in agents:
            reward = 0.0
            try:
                car = state.cars[agent]
                if car.ball_touches > 0:
                    car_pos = self._car_pos(state, agent)
                    if is_in_air(car_pos):
                        reward = self.aerial_multiplier
                    else:
                        reward = 1.0
            except Exception:
                pass
            rewards[agent] = reward
        return rewards
