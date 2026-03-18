"""
SaveReward: reward for deflecting the ball away from own goal.
Detected when:
  - Ball is within a threshold distance of own goal AND moving toward it
  - After a touch, ball is moving AWAY from own goal
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import vec3_normalize, BLUE_GOAL_POS, ORANGE_GOAL_POS


class SaveReward(BaseReward):
    weight: float = 1.0
    active_from_stage: int = 2

    SAVE_ZONE_DIST   = 2000.0   # uu from own goal to be in "save zone"
    MIN_DEFLECT_SPEED = 300.0   # uu/s

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        ball_pos = self._ball_pos(state)
        ball_vel = self._ball_vel(state)
        rewards: Dict[Any, float] = {}

        for agent in agents:
            reward = 0.0
            try:
                team      = self._team(state, agent)
                own_goal  = BLUE_GOAL_POS if team == 0 else ORANGE_GOAL_POS
                to_goal   = own_goal - ball_pos
                dist      = float(np.linalg.norm(to_goal))

                if dist < self.SAVE_ZONE_DIST:
                    # Ball moving toward our goal = danger
                    to_goal_dir = vec3_normalize(to_goal)
                    vel_toward  = float(np.dot(ball_vel, to_goal_dir))
                    # After a touch, if ball is now moving AWAY, reward the save
                    if vel_toward < -self.MIN_DEFLECT_SPEED:
                        # The more it's deflected away, the higher the reward
                        reward = min(abs(vel_toward) / 2000.0, 1.0)
            except Exception:
                pass
            rewards[agent] = reward

        return rewards
