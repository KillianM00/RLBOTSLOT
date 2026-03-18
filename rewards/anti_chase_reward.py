"""
AntiChaseReward: penalise mindless ball-chasing.

Penalties applied when:
  1. Bot touches the ball twice in a row with no meaningful progress
     (ball didn't advance toward opponent goal between touches).
  2. Bot is rushing toward ball when a teammate is already closer
     (in multi-agent stages).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import ORANGE_GOAL_POS, BLUE_GOAL_POS, own_goal_pos, goal_pos_for_team


class AntiChaseReward(BaseReward):
    weight: float = 0.8
    active_from_stage: int = 0

    SAME_TOUCH_PENALTY    = cfg.ANTI_CHASE_SAME_TOUCH_PENALTY
    OVERCOMMIT_PENALTY    = cfg.ANTI_CHASE_OVERCOMMIT_PENALTY
    NO_PROGRESS_THRESHOLD = 200.0   # uu; ball must advance at least this much

    def __init__(self) -> None:
        self._last_touch_tick: Dict[Any, float] = {}
        self._last_ball_pos_at_touch: Dict[Any, Optional[np.ndarray]] = {}

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        self._last_touch_tick         = {a: -1.0 for a in agents}
        self._last_ball_pos_at_touch  = {a: None  for a in agents}

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

        tick = getattr(state, "tick_count", 0)
        for agent in agents:
            reward = 0.0
            try:
                car = state.cars[agent]
                if car.ball_touches > 0:
                    prev_bpos = self._last_ball_pos_at_touch.get(agent)
                    if prev_bpos is not None:
                        team     = self._team(state, agent)
                        att_goal = ORANGE_GOAL_POS if team == 0 else BLUE_GOAL_POS
                        old_dist = float(np.linalg.norm(att_goal - prev_bpos))
                        new_dist = float(np.linalg.norm(att_goal - ball_pos))
                        if (old_dist - new_dist) < self.NO_PROGRESS_THRESHOLD:
                            reward = self.SAME_TOUCH_PENALTY
                    self._last_touch_tick[agent]        = float(tick)
                    self._last_ball_pos_at_touch[agent] = ball_pos.copy()
            except Exception:
                pass
            rewards[agent] = reward

        return rewards
