"""
env_factory.build_env() is the entry point called by each rlgym-ppo worker process.
Returns a fully configured RLGym environment backed by RocketSim.
"""
from __future__ import annotations

import config as cfg
from engine.curriculum import CurriculumScheduler
from engine.obs_builder import CustomObsBuilder
from engine.state_mutators import CurriculumStateMutator
from engine.done_conditions import build_termination_condition, build_truncation_condition

try:
    from rlgym.api import RLGym                                         # type: ignore
    from rlgym.rocket_league.action_parsers import (                    # type: ignore
        LookupTableAction, RepeatAction,
    )
    from rlgym.rocket_league.sim import RocketSimEngine                 # type: ignore
    _RLGYM_AVAILABLE = True
except ImportError:
    _RLGYM_AVAILABLE = False

# Shared scheduler instance lives in the main process;
# worker processes receive stage as an integer argument.
_global_scheduler: CurriculumScheduler | None = None


def get_global_scheduler() -> CurriculumScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = CurriculumScheduler()
    return _global_scheduler


def build_env(curriculum_stage: int = 0, tick_skip: int = cfg.TICK_SKIP):
    """
    Construct and return an rlgym RLGym environment for the given curriculum stage.
    Called once per worker process (multiprocessing fork).

    Parameters
    ----------
    curriculum_stage : int
        Current curriculum stage (0–4).  Workers receive this from the main process.
    tick_skip : int
        Number of physics ticks per bot action (default cfg.TICK_SKIP = 8).

    Returns
    -------
    RLGym instance (or None if rlgym is not installed, for testing).
    """
    if not _RLGYM_AVAILABLE:
        return None

    # Import reward builder here to avoid circular imports at module level
    from rewards.combined_reward import DynamicCombinedReward
    from engine.gym_wrapper import GymWrapper

    rlgym_env = RLGym(
        state_mutator=CurriculumStateMutator(stage=curriculum_stage),
        obs_builder=CustomObsBuilder(),
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=DynamicCombinedReward.from_config(curriculum_stage),
        transition_engine=RocketSimEngine(),
        termination_cond=build_termination_condition(),
        truncation_cond=build_truncation_condition(),
    )
    return GymWrapper(rlgym_env)
