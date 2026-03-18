"""
RankTracker: estimates the bot's current competitive rank from self-play statistics.

Uses a simple Elo-like calculation over a rolling window of recent episodes:
  - Win rate > 0.6 → promote one tier
  - Win rate < 0.35 → demote one tier
  - Otherwise stay

Starts at cfg.STARTING_RANK and evolves as training progresses.
"""
from __future__ import annotations

from collections import deque
from typing import Deque

import config as cfg
from utils.rank_enum import RankTier


class RankTracker:
    """
    Tracks bot win/loss results to estimate its rank tier.
    """

    WINDOW_SIZE      = 100   # episodes in rolling window
    PROMOTE_THRESH   = 0.60  # win-rate above this → promote
    DEMOTE_THRESH    = 0.35  # win-rate below this → demote
    MIN_WINDOW       = 30    # minimum episodes before rank can change

    def __init__(self) -> None:
        self.current_tier: RankTier = RankTier.from_string(cfg.STARTING_RANK)
        self._results: Deque[int] = deque(maxlen=self.WINDOW_SIZE)  # 1=win, 0=loss

    def record_result(self, win: bool) -> None:
        self._results.append(1 if win else 0)
        self._maybe_update_rank()

    def _maybe_update_rank(self) -> None:
        if len(self._results) < self.MIN_WINDOW:
            return
        win_rate = sum(self._results) / len(self._results)
        if win_rate > self.PROMOTE_THRESH:
            self.current_tier = RankTier(min(
                int(self.current_tier) + 1,
                int(RankTier.SUPERSONIC_LEGEND),
            ))
            self._results.clear()   # reset window after rank change
        elif win_rate < self.DEMOTE_THRESH:
            self.current_tier = RankTier(max(int(self.current_tier) - 1, 0))
            self._results.clear()

    @property
    def win_rate(self) -> float:
        if not self._results:
            return 0.5
        return sum(self._results) / len(self._results)

    def state_dict(self) -> dict:
        return {
            "tier": int(self.current_tier),
            "results": list(self._results),
        }

    def load_state_dict(self, d: dict) -> None:
        self.current_tier = RankTier(d["tier"])
        self._results = deque(d.get("results", []), maxlen=self.WINDOW_SIZE)
