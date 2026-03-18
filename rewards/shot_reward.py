"""
ShotReward: reward when ball is struck toward opponent goal with speed.
Heuristic: after a touch, ball velocity toward opponent goal > threshold.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import vec3_normalize, ORANGE_GOAL_POS, BLUE_GOAL_POS


class ShotReward(BaseReward):
    weight: float = 1.0
    active_from_stage: int = 1

    MIN_BALL_SPEED    = 800.0   # uu/s
    GOAL_ALIGN_THRESH = 0.7     # cosine similarity of ball vel to goal direction

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        ball_vel = self._ball_vel(state)
        ball_speed = float(np.linalg.norm(ball_vel))
        rewards: Dict[Any, float] = {}

        for agent in agents:
            reward = 0.0
            try:
                if ball_speed < self.MIN_BALL_SPEED:
                    rewards[agent] = 0.0
                    continue
                team     = self._team(state, agent)
                goal_pos = ORANGE_GOAL_POS if team == 0 else BLUE_GOAL_POS
                ball_pos = self._ball_pos(state)
                to_goal  = vec3_normalize(goal_pos - ball_pos)
                ball_dir = ball_vel / (ball_speed + 1e-8)
                align    = float(np.dot(ball_dir, to_goal))
                if align > self.GOAL_ALIGN_THRESH:
                    # Scale reward by alignment and speed (normalised)
                    reward = align * min(ball_speed / 2000.0, 1.0)
            except Exception:
                pass
            rewards[agent] = reward

        return rewards
