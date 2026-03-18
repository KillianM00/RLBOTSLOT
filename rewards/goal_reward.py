"""
GoalScoredReward: +GOAL_REWARD for scoring, -GOAL_REWARD for conceding.
Uses rlgym's scored event from game state if available; otherwise monitors
ball crossing the goal line.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import FIELD_LENGTH, GOAL_WIDTH, GOAL_HEIGHT, own_goal_pos


class GoalScoredReward(BaseReward):
    weight: float = 5.0
    active_from_stage: int = 0

    def __init__(
        self,
        goal_reward: float = cfg.GOAL_REWARD,
        concede_reward: float = cfg.CONCEDE_REWARD,
    ) -> None:
        self.goal_reward    = goal_reward
        self.concede_reward = concede_reward
        self._prev_scores: Dict[Any, int] = {}

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        pass

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        # rlgym 2.x: state.goal_scored (bool), state.scoring_team (0=blue, 1=orange)
        rewards: Dict[Any, float] = {}
        goal_scored = getattr(state, "goal_scored", False)
        scoring_team = getattr(state, "scoring_team", -1)
        for agent in agents:
            reward = 0.0
            if goal_scored:
                try:
                    my_team = self._team(state, agent)
                    if my_team == scoring_team:
                        reward = self.goal_reward
                    else:
                        reward = self.concede_reward
                except Exception:
                    pass
            rewards[agent] = reward
        return rewards
