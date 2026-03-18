"""
DynamicCombinedReward: weighted aggregation of all 13 reward components.

Weights are adjusted at each curriculum stage via update_weights().
Components whose active_from_stage > current_stage return 0 automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import config as cfg
from rewards.base_reward import BaseReward
from rewards.touch_reward import TouchReward
from rewards.ball_control_reward import BallControlReward
from rewards.goal_reward import GoalScoredReward
from rewards.shot_reward import ShotReward
from rewards.save_reward import SaveReward
from rewards.boost_reward import BoostEfficiencyReward
from rewards.speed_reward import SpeedReward
from rewards.positioning_reward import PositioningReward
from rewards.aerial_reward import AerialReward
from rewards.flick_reward import FlickReward
from rewards.demo_reward import DemoReward
from rewards.anti_chase_reward import AntiChaseReward
from rewards.curriculum_progress_reward import CurriculumProgressReward
from rewards.concept_reward import ConceptDebateReward
from rewards.ball_proximity_reward import BallProximityReward


_ALL_COMPONENTS: Dict[str, type] = {
    "TouchReward":              TouchReward,
    "BallControlReward":        BallControlReward,
    "GoalScoredReward":         GoalScoredReward,
    "ShotReward":               ShotReward,
    "SaveReward":               SaveReward,
    "BoostEfficiencyReward":    BoostEfficiencyReward,
    "SpeedReward":              SpeedReward,
    "PositioningReward":        PositioningReward,
    "AerialReward":             AerialReward,
    "FlickReward":              FlickReward,
    "DemoReward":               DemoReward,
    "AntiChaseReward":          AntiChaseReward,
    "CurriculumProgressReward": CurriculumProgressReward,
    "ConceptDebateReward":      ConceptDebateReward,
    "BallProximityReward":      BallProximityReward,
}


class DynamicCombinedReward:
    """
    Orchestrates all reward components with dynamic weighting.

    Compatible with rlgym RewardFunction interface.
    """

    def __init__(
        self,
        stage: int = 0,
        concept_debate_reward: Optional[ConceptDebateReward] = None,
    ) -> None:
        self.stage = stage
        self._components: Dict[str, BaseReward] = {}

        # Instantiate all components
        for name, cls in _ALL_COMPONENTS.items():
            if name == "ConceptDebateReward":
                self._components[name] = concept_debate_reward or ConceptDebateReward()
            else:
                self._components[name] = cls()

        # Load weights for the initial stage
        self._weights: Dict[str, float] = {}
        self.update_weights(stage)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        stage: int = 0,
        concept_debate_reward: Optional[ConceptDebateReward] = None,
    ) -> "DynamicCombinedReward":
        return cls(stage=stage, concept_debate_reward=concept_debate_reward)

    # ------------------------------------------------------------------
    def update_weights(self, stage: int) -> None:
        """Switch to the weight preset for the given curriculum stage."""
        self.stage   = stage
        preset       = cfg.REWARD_WEIGHTS.get(stage, cfg.REWARD_WEIGHTS[0])
        self._weights = {name: 0.0 for name in _ALL_COMPONENTS}
        self._weights.update(preset)

    def get_component(self, name: str) -> Optional[BaseReward]:
        return self._components.get(name)

    # ------------------------------------------------------------------
    # rlgym RewardFunction interface
    # ------------------------------------------------------------------

    def reset(self, agents: List, initial_state: Any, shared_info: Dict) -> None:
        for comp in self._components.values():
            try:
                comp.reset(agents, initial_state, shared_info)
            except Exception:
                pass

    def get_rewards(
        self,
        agents: List,
        state: Any,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: Dict,
    ) -> Dict[Any, float]:
        combined: Dict[Any, float] = {a: 0.0 for a in agents}

        for name, comp in self._components.items():
            w = self._weights.get(name, 0.0)
            if w == 0.0:
                continue
            try:
                step_rewards = comp.get_rewards(
                    agents, state, is_terminated, is_truncated, shared_info
                )
                for agent in agents:
                    combined[agent] += w * step_rewards.get(agent, 0.0)
            except Exception:
                pass

        return combined

    # ------------------------------------------------------------------
    # Convenience: curriculum progress notification
    # ------------------------------------------------------------------

    def notify_stage_advance(self, new_stage: int) -> None:
        self.update_weights(new_stage)
        prog_r = self._components.get("CurriculumProgressReward")
        if prog_r and hasattr(prog_r, "notify_stage_advance"):
            prog_r.notify_stage_advance(new_stage)
