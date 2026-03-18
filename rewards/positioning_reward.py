"""
PositioningReward: reward proper field positioning and rotation.
Penalises over-committing (being too close to ball when not possession holder)
and rewards staying in a good defensive/transition position.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import FIELD_LENGTH, own_goal_pos, goal_pos_for_team


class PositioningReward(BaseReward):
    weight: float = 0.5
    active_from_stage: int = 2

    # When not closest car to ball, should be between ball and own goal
    ROTATION_REWARD_SCALE = 0.3
    OVERCOMMIT_DIST       = 300.0   # uu; if within this of ball, penalise if not touch

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        ball_pos = self._ball_pos(state)
        rewards: Dict[Any, float] = {}

        for agent in agents:
            reward = 0.0
            try:
                car_pos  = self._car_pos(state, agent)
                team     = self._team(state, agent)
                own_g    = own_goal_pos(team == 0)

                # Rotation reward: car between ball and own goal
                to_ball  = ball_pos - own_g
                to_car   = car_pos  - own_g
                dist_ball = float(np.linalg.norm(to_ball))
                dist_car  = float(np.linalg.norm(to_car))

                if dist_ball > 100 and dist_car < dist_ball:
                    # Car is closer to own goal than ball — good rotation
                    reward = self.ROTATION_REWARD_SCALE * (
                        1.0 - dist_car / (dist_ball + 1.0)
                    )
                elif dist_car > dist_ball:
                    # Car is further back than ball — slight bonus for goal-side position
                    reward = 0.05

            except Exception:
                pass
            rewards[agent] = reward

        return rewards
