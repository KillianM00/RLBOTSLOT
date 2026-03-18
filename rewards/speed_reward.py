"""
SpeedReward: reward ball velocity directed toward the opponent's goal.
Encourages the bot to hit the ball hard and in the right direction.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import vec3_normalize, ORANGE_GOAL_POS, BLUE_GOAL_POS, MAX_BALL_SPEED


class SpeedReward(BaseReward):
    weight: float = 0.3
    active_from_stage: int = 2

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        ball_vel  = self._ball_vel(state)
        ball_pos  = self._ball_pos(state)
        ball_speed = float(np.linalg.norm(ball_vel))
        rewards: Dict[Any, float] = {}

        for agent in agents:
            reward = 0.0
            try:
                team     = self._team(state, agent)
                goal_pos = ORANGE_GOAL_POS if team == 0 else BLUE_GOAL_POS
                to_goal  = vec3_normalize(goal_pos - ball_pos)
                if ball_speed > 100.0:
                    ball_dir   = ball_vel / (ball_speed + 1e-8)
                    alignment  = float(np.dot(ball_dir, to_goal))
                    # Only reward positive alignment (ball heading toward goal)
                    reward = max(alignment, 0.0) * (ball_speed / MAX_BALL_SPEED)
            except Exception:
                pass
            rewards[agent] = reward

        return rewards
