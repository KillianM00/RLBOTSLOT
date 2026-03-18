"""
Custom observation builder extending rlgym's DefaultObs.
Adds:
  - Normalised time remaining in episode
  - Active-concept embedding (from ProfessorPolicy gating) injected externally
  - Extra derived features: ball-to-goal distance, car-to-ball distance, etc.

Total obs dimension: ~107 base + optional concept embedding (128)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rlgym.rocket_league.obs_builders import DefaultObs  # type: ignore
    from rlgym.rocket_league.api import GameState  # type: ignore
    _RLGYM_AVAILABLE = True
except ImportError:
    _RLGYM_AVAILABLE = False
    DefaultObs = object  # fallback for import-time

import config as cfg
from utils.math_utils import (
    vec3_norm, vec3_normalize,
    ORANGE_GOAL_POS, BLUE_GOAL_POS,
)


class CustomObsBuilder(DefaultObs if _RLGYM_AVAILABLE else object):
    """
    Extends rlgym DefaultObs with extra derived features.
    If rlgym is not installed, behaves as a no-op (for unit testing).
    """

    EXTRA_DIM = 6  # number of additional features appended

    def __init__(self, inject_concept_embed: bool = False) -> None:
        if _RLGYM_AVAILABLE:
            super().__init__(zero_padding=None)
        self.inject_concept_embed = inject_concept_embed
        # Set externally by ProfessorPolicy after gating step
        self._concept_embed: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def set_concept_embed(self, embed: np.ndarray) -> None:
        self._concept_embed = embed

    # ------------------------------------------------------------------
    def build_obs(self, agents, state, shared_info: dict) -> Dict:
        if not _RLGYM_AVAILABLE:
            return {}
        base_obs = super().build_obs(agents, state, shared_info)

        for agent_id, obs in base_obs.items():
            extra = self._compute_extra(agent_id, state)
            obs = np.concatenate([obs, extra])
            if self.inject_concept_embed and self._concept_embed is not None:
                obs = np.concatenate([obs, self._concept_embed])
            base_obs[agent_id] = obs.astype(np.float32)

        return base_obs

    def _compute_extra(self, agent_id, state) -> np.ndarray:
        """Compute 6 extra normalised features."""
        try:
            car   = state.cars[agent_id]
            ball  = state.ball
            team  = car.team_num   # 0=blue, 1=orange

            car_pos  = np.asarray(car.physics.position,  dtype=np.float32)
            ball_pos = np.asarray(ball.position,          dtype=np.float32)

            goal_pos = ORANGE_GOAL_POS if team == 0 else BLUE_GOAL_POS
            own_goal = BLUE_GOAL_POS   if team == 0 else ORANGE_GOAL_POS

            car_to_ball     = vec3_norm(ball_pos - car_pos) / 10000.0
            ball_to_goal    = vec3_norm(goal_pos - ball_pos) / 10000.0
            ball_to_own_gl  = vec3_norm(own_goal - ball_pos) / 10000.0
            car_to_goal     = vec3_norm(goal_pos - car_pos)  / 10000.0
            is_aerial       = float(car_pos[2] > 20.0)
            ball_is_aerial  = float(ball_pos[2] > 150.0)

            return np.array([
                car_to_ball, ball_to_goal, ball_to_own_gl,
                car_to_goal, is_aerial, ball_is_aerial,
            ], dtype=np.float32)
        except Exception:
            return np.zeros(6, dtype=np.float32)
