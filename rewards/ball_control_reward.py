"""
BallControlReward: rewards maintaining possession (dribbling).
Dribbling is detected when:
  - Ball z-position is between ~95 (on roof) and ~200 (in air above car)
  - Car z-velocity is positive (rising with the ball)
  - Ball and car positions are close (< 200 uu horizontal)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward


class BallControlReward(BaseReward):
    """
    Continuous reward for keeping the ball close to the car
    and elevated (dribble position).
    Max: 1.0 per tick when in perfect dribble; 0 when ball is far away.
    """

    weight: float = 1.0
    active_from_stage: int = 1

    DRIBBLE_Z_MIN   = 92.75    # ball radius on ground
    DRIBBLE_Z_MAX   = 300.0    # max meaningful dribble height
    MAX_HORIZ_DIST  = 200.0    # uu, horizontal proximity threshold

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
                car_pos = self._car_pos(state, agent)
                horiz_dist = float(np.linalg.norm(ball_pos[:2] - car_pos[:2]))
                ball_z     = ball_pos[2]
                car_z      = car_pos[2]

                if (
                    self.DRIBBLE_Z_MIN < ball_z < self.DRIBBLE_Z_MAX
                    and horiz_dist < self.MAX_HORIZ_DIST
                    and ball_z > car_z
                ):
                    # Reward inversely proportional to horizontal distance
                    reward = 1.0 - horiz_dist / self.MAX_HORIZ_DIST
            except Exception:
                pass
            rewards[agent] = float(reward)

        return rewards
