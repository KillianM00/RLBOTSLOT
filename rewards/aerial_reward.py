"""
AerialReward: reward for executing aerial manoeuvres.
  +reward for touching ball while in the air
  +bonus for ball being high (aerial = meaningful)
  Scales with ball height to distinguish ceiling shots from low aerials.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import is_in_air, CEILING


class AerialReward(BaseReward):
    weight: float = 1.0
    active_from_stage: int = 3

    MIN_BALL_HEIGHT   = 300.0    # uu; below this is a ground hit, not aerial
    MIN_CAR_HEIGHT    = 100.0    # uu; car must also be in the air

    def __init__(self) -> None:
        self._last_touch_time: Dict[Any, float] = {}

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        self._last_touch_time = {a: -1.0 for a in agents}

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        ball_pos = self._ball_pos(state)
        ball_z   = ball_pos[2]
        rewards: Dict[Any, float] = {}

        for agent in agents:
            reward = 0.0
            try:
                car_pos = self._car_pos(state, agent)
                car_z   = car_pos[2]

                # Both car and ball must be meaningfully in the air
                if car_z > self.MIN_CAR_HEIGHT and ball_z > self.MIN_BALL_HEIGHT:
                    # Check for a fresh touch
                    touch = getattr(state.ball, "last_touch", None)
                    if touch is not None:
                        touch_time = getattr(touch, "game_seconds", -1.0)
                        prev_time  = self._last_touch_time.get(agent, -1.0)
                        is_my_touch = getattr(touch, "player_index", -1) == int(
                            agent.split("_")[-1] if isinstance(agent, str) else agent
                        )
                        if is_my_touch and touch_time != prev_time:
                            self._last_touch_time[agent] = touch_time
                            # Scale reward with ball height
                            height_factor = min(ball_z / CEILING, 1.0)
                            reward = 0.5 + 0.5 * height_factor
            except Exception:
                pass
            rewards[agent] = reward

        return rewards
