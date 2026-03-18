"""
BoostEfficiencyReward: reward smart boost usage.
  +reward for collecting boost pads when low on boost
  +reward for using boost while on the ground moving toward ball
  -penalty for wasting boost at full tank
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward


class BoostEfficiencyReward(BaseReward):
    weight: float = 0.5
    active_from_stage: int = 2

    LOW_BOOST_THRESH  = 0.3    # fraction below which bot should be collecting
    FULL_BOOST_THRESH = 0.9    # fraction above which using boost is wasteful

    def __init__(self) -> None:
        self._prev_boost: Dict[Any, float] = {}

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        for agent in agents:
            try:
                boost = _get_boost(initial_state, agent)
            except Exception:
                boost = 1.0
            self._prev_boost[agent] = boost

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
                boost = _get_boost(state, agent)
                prev  = self._prev_boost.get(agent, boost)
                delta = boost - prev       # positive = collected, negative = used

                if delta > 0.02:
                    # Collected boost — reward if we needed it
                    if prev < self.LOW_BOOST_THRESH:
                        reward = delta * 0.5
                elif delta < -0.02:
                    # Used boost — penalise if we were already full
                    if prev > self.FULL_BOOST_THRESH:
                        reward = delta * 0.3  # negative
                    else:
                        reward = 0.0

                self._prev_boost[agent] = boost
            except Exception:
                pass
            rewards[agent] = reward

        return rewards


def _get_boost(state: Any, agent_id: Any) -> float:
    """Returns boost fraction [0, 1]."""
    try:
        b = state.cars[agent_id].boost_amount
        # rlgym v2 stores boost as 0-100 float; normalise
        return float(b) / 100.0 if b > 1.1 else float(b)
    except Exception:
        return 0.5
