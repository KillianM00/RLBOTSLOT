"""
DemoReward: reward for demolishing an opponent strategically.
Demos are only rewarded when the ball is near the opponent (so it's disruptive).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward


class DemoReward(BaseReward):
    weight: float = 0.3
    active_from_stage: int = 3

    DEMO_REWARD = 2.0

    def __init__(self) -> None:
        self._prev_demo_counts: Dict[Any, int] = {}

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        for agent in agents:
            try:
                self._prev_demo_counts[agent] = _get_demo_count(initial_state, agent)
            except Exception:
                self._prev_demo_counts[agent] = 0

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
                current = _get_demo_count(state, agent)
                prev    = self._prev_demo_counts.get(agent, current)
                if current > prev:
                    # New demo — reward if ball is close to opponent
                    ball_pos = self._ball_pos(state)
                    opp_pos  = _nearest_opponent_pos(state, agent)
                    if opp_pos is not None:
                        dist = float(np.linalg.norm(ball_pos - opp_pos))
                        if dist < cfg.DEMO_MIN_DISTANCE:
                            reward = self.DEMO_REWARD
                self._prev_demo_counts[agent] = current
            except Exception:
                pass
            rewards[agent] = reward

        return rewards


def _get_demo_count(state: Any, agent_id: Any) -> int:
    try:
        return int(state.cars[agent_id].match_demolishes)
    except Exception:
        return 0


def _nearest_opponent_pos(state: Any, agent_id: Any):
    try:
        my_team = state.cars[agent_id].team_num
        for cid, car in state.cars.items():
            if cid != agent_id and car.team_num != my_team:
                c = car.physics.position
                return np.array([c.x, c.y, c.z], dtype=np.float32)
    except Exception:
        pass
    return None
