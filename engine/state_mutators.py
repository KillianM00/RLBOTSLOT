"""
Curriculum-aware state mutators for rlgym environments.
Each stage can have different state initialization strategies.
"""
from __future__ import annotations

import random
import numpy as np

try:
    from rlgym.rocket_league.state_mutators import (  # type: ignore
        MutatorSequence, KickoffMutator, FixedTeamSizeMutator,
    )
    from rlgym.api import StateMutator                           # type: ignore
    from rlgym.rocket_league.api import GameState               # type: ignore
    _RLGYM_AVAILABLE = True
except ImportError:
    _RLGYM_AVAILABLE = False
    StateMutator = object

from engine.curriculum import STAGES


class CurriculumStateMutator(StateMutator if _RLGYM_AVAILABLE else object):
    """
    Selects an appropriate state mutator sequence for the current curriculum stage.

    Stage 0 (Kickoffs): Standard kickoff positions
    Stage 1 (Possession): Random ball placement in attacking half
    Stage 2+ (Full): Mixed kickoff + random ball
    """

    def __init__(self, stage: int = 0) -> None:
        self.stage = stage

    def apply(self, state: "GameState", shared_info: dict) -> None:
        if not _RLGYM_AVAILABLE:
            return

        # Always ensure cars exist first
        FixedTeamSizeMutator(blue_size=1, orange_size=1).apply(state, shared_info)

        if self.stage == 0:
            KickoffMutator().apply(state, shared_info)
        elif self.stage == 1:
            KickoffMutator().apply(state, shared_info)
            self._place_ball_random_attacking(state)
        else:
            if random.random() < 0.5:
                KickoffMutator().apply(state, shared_info)
            else:
                KickoffMutator().apply(state, shared_info)
                self._place_ball_random_attacking(state)

    def _place_ball_random_attacking(self, state: "GameState") -> None:
        """Place the ball randomly in the attacking half (uses numpy arrays like rlgym expects)."""
        try:
            x = random.uniform(-3500, 3500)
            y = random.uniform(0, 4500)   # orange half
            z = random.uniform(93, 500)   # slightly elevated
            state.ball.position       = np.array([x, y, z], dtype=np.float32)
            state.ball.linear_velocity  = np.zeros(3, dtype=np.float32)
            state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        except AttributeError:
            pass
