"""
CurriculumScheduler: tracks training progression through 5 stages.
Each stage unlocks more complex reward components and match configurations.

Stage 0 – Kickoffs only, master basic ball contact
Stage 1 – 1v0 possession and shooting
Stage 2 – 1v1 defence introduced
Stage 3 – Aerials, flicks, advanced mechanics
Stage 4 – Full match with all rewards
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import config as cfg


@dataclass
class TrainingMetrics:
    """Rolling metrics window used by CurriculumScheduler.should_advance()."""
    total_timesteps: int = 0
    goal_rate: float = 0.0          # goals per episode
    avg_reward: float = 0.0
    concept_win_rates: Dict[str, float] = field(default_factory=dict)

    def update(
        self,
        new_timesteps: int,
        episode_goals: float,
        episode_reward: float,
    ) -> None:
        self.total_timesteps += new_timesteps
        # Exponential moving average
        alpha = 0.05
        self.goal_rate  = (1 - alpha) * self.goal_rate  + alpha * episode_goals
        self.avg_reward = (1 - alpha) * self.avg_reward + alpha * episode_reward


@dataclass
class CurriculumStage:
    stage_id: int
    name: str
    min_timesteps: int
    goal_rate_threshold: float
    team_size: int = 1          # cars per side
    enable_opponent: bool = False


# Predefined stages
STAGES = [
    CurriculumStage(0, "Kickoffs",       0,           0.0,  1, False),
    CurriculumStage(1, "Possession",     5_000_000,   0.3,  1, False),
    CurriculumStage(2, "Defence",        50_000_000,  0.4,  1, True),
    CurriculumStage(3, "Mechanics",      200_000_000, 0.45, 1, True),
    CurriculumStage(4, "FullMatch",      500_000_000, 0.5,  2, True),
]


class CurriculumScheduler:
    """
    Manages stage transitions based on training metrics.
    Stage can only advance (never regress).
    """

    def __init__(self) -> None:
        self.stages = STAGES
        self.current_stage: int = 0
        self._stage_start_time: float = time.time()
        self._advance_callbacks: list = []

    # ------------------------------------------------------------------
    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.current_stage]

    # ------------------------------------------------------------------
    def should_advance(self, metrics: TrainingMetrics) -> bool:
        if self.current_stage >= len(self.stages) - 1:
            return False
        next_stage = self.stages[self.current_stage + 1]
        ts_ok   = metrics.total_timesteps >= next_stage.min_timesteps
        rate_ok = metrics.goal_rate       >= next_stage.goal_rate_threshold
        return ts_ok and rate_ok

    def advance(self) -> int:
        """Advance to the next stage. Returns the new stage id."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self._stage_start_time = time.time()
            for cb in self._advance_callbacks:
                cb(self.current_stage)
        return self.current_stage

    def register_advance_callback(self, fn) -> None:
        """Register a callable invoked with new_stage_id when stage advances."""
        self._advance_callbacks.append(fn)

    def get_reward_weights(self) -> Dict[str, float]:
        return cfg.REWARD_WEIGHTS.get(self.current_stage, {})

    def state_dict(self) -> dict:
        return {"current_stage": self.current_stage}

    def load_state_dict(self, d: dict) -> None:
        self.current_stage = d["current_stage"]
