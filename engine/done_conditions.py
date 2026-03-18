"""
Custom terminal and truncation conditions.
"""
from __future__ import annotations

from typing import Dict, List

try:
    from rlgym.rocket_league.done_conditions import (  # type: ignore
        GoalCondition, TimeoutCondition, AnyCondition,
    )
    from rlgym.api import DoneCondition                           # type: ignore
    from rlgym.rocket_league.api import GameState                 # type: ignore
    _RLGYM_AVAILABLE = True
except ImportError:
    _RLGYM_AVAILABLE = False
    DoneCondition = object

import config as cfg


class NoTouchTimeoutCondition(DoneCondition if _RLGYM_AVAILABLE else object):
    """
    Truncates an episode if the ball has not been touched for `timeout_seconds`.
    Encourages the bot to engage with the ball.
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._last_touch_time: Dict = {}
        self._game_time: Dict = {}

    def reset(self, agents: List, initial_state, shared_info: dict) -> None:
        for agent in agents:
            self._last_touch_time[agent] = 0.0
            self._game_time[agent] = 0.0

    def is_done(self, agents: List, state, shared_info: dict) -> Dict:
        result = {}
        try:
            elapsed = getattr(state, "game_seconds_elapsed", 0.0)
            last_touch = getattr(state.ball, "last_touch", None)
            touch_time = getattr(last_touch, "game_seconds", 0.0) if last_touch else 0.0

            for agent in agents:
                time_since_touch = elapsed - touch_time
                result[agent] = time_since_touch > self.timeout_seconds
        except Exception:
            for agent in agents:
                result[agent] = False
        return result


def build_termination_condition():
    """Goal scored ends the episode."""
    if not _RLGYM_AVAILABLE:
        return None
    return AnyCondition(
        GoalCondition(),
        TimeoutCondition(cfg.MAX_EPISODE_SECONDS),
    )


def build_truncation_condition():
    """No-touch timeout truncates the episode."""
    if not _RLGYM_AVAILABLE:
        return None
    return NoTouchTimeoutCondition(cfg.NO_TOUCH_TIMEOUT)
