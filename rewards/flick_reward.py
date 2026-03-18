"""
FlickReward: reward for executing a flick.

A flick is characterised by:
  1. Ball is on the roof of the car (ball z ≈ car z + ~155 uu = car height + ball radius)
  2. Then, within the next few frames, ball velocity has a large component
     perpendicular to the car's forward vector (lateral kick off)

Detection heuristic:
  - Track ball-on-roof state
  - When ball leaves roof, check if perpendicular velocity delta exceeds threshold
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import config as cfg
from rewards.base_reward import BaseReward
from utils.math_utils import vec3_norm, vec3_normalize, forward_vec


CAR_HEIGHT        = 36.16   # uu
BALL_RADIUS       = 91.25   # uu
ROOF_Z_OFFSET     = CAR_HEIGHT + BALL_RADIUS  # ~127 uu above car centre


class FlickReward(BaseReward):
    weight: float = 0.8
    active_from_stage: int = 3

    ROOF_TOLERANCE    = 50.0   # uu; ball z within this of expected roof height
    FLICK_VEL_THRESH  = cfg.FLICK_VELOCITY_THRESHOLD  # uu/s lateral delta

    def __init__(self) -> None:
        self._on_roof: Dict[Any, bool] = {}
        self._prev_ball_vel: Optional[np.ndarray] = None

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        self._on_roof      = {a: False for a in agents}
        self._prev_ball_vel = None

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        ball_pos  = self._ball_pos(state)
        ball_vel  = self._ball_vel(state)
        rewards: Dict[Any, float] = {}

        for agent in agents:
            reward = 0.0
            try:
                car_pos = self._car_pos(state, agent)
                horiz_dist = float(np.linalg.norm(ball_pos[:2] - car_pos[:2]))
                expected_z = car_pos[2] + ROOF_Z_OFFSET

                currently_on_roof = (
                    abs(ball_pos[2] - expected_z) < self.ROOF_TOLERANCE
                    and horiz_dist < 120.0
                )

                was_on_roof = self._on_roof.get(agent, False)

                if was_on_roof and not currently_on_roof and self._prev_ball_vel is not None:
                    # Ball just left the roof — check for lateral velocity spike
                    try:
                        car = state.cars[agent]
                        pitch = getattr(car.physics.rotation, "pitch", 0.0)
                        yaw   = getattr(car.physics.rotation, "yaw",   0.0)
                        fwd   = forward_vec(pitch, yaw)
                        # Perpendicular velocity = ball_vel - (ball_vel · fwd) * fwd
                        along = np.dot(ball_vel, fwd) * fwd
                        perp  = ball_vel - along
                        perp_speed = vec3_norm(perp)
                        if perp_speed > self.FLICK_VEL_THRESH:
                            reward = min(perp_speed / 2000.0, 1.0)
                    except Exception:
                        pass

                self._on_roof[agent] = currently_on_roof
            except Exception:
                pass
            rewards[agent] = reward

        self._prev_ball_vel = ball_vel.copy()
        return rewards
