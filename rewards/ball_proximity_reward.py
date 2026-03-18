"""
BallProximityReward: small dense reward for being close to the ball.
Fires every step — provides gradient signal during sparse early training.
Reward = 1 - clamp(dist / MAX_DIST, 0, 1)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from rewards.base_reward import BaseReward

MAX_DIST = 5000.0  # unreal units; roughly half the field diagonal


class BallProximityReward(BaseReward):
    weight: float = 0.1
    active_from_stage: int = 0

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
            try:
                car_pos = self._car_pos(state, agent)
                dist = float(np.linalg.norm(ball_pos - car_pos))
                rewards[agent] = 1.0 - min(dist / MAX_DIST, 1.0)
            except Exception:
                rewards[agent] = 0.0
        return rewards
