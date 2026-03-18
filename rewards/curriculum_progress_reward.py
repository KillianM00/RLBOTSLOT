"""
CurriculumProgressReward: one-time bonus when the curriculum stage advances.
Set to 0 in Stage 4 (no more stages to unlock).
"""
from __future__ import annotations

from typing import Any, Dict, List

import config as cfg
from rewards.base_reward import BaseReward


class CurriculumProgressReward(BaseReward):
    weight: float = 1.0
    active_from_stage: int = 0

    STAGE_BONUS = 50.0   # large one-time reward for advancing a stage

    def __init__(self) -> None:
        self._prev_stage: int = 0
        self._bonus_pending: bool = False

    def notify_stage_advance(self, new_stage: int) -> None:
        """Called externally by CurriculumScheduler callback."""
        if new_stage > self._prev_stage:
            self._prev_stage    = new_stage
            self._bonus_pending = True

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        if self._bonus_pending:
            self._bonus_pending = False
            return {a: self.STAGE_BONUS for a in agents}
        return {a: 0.0 for a in agents}
