"""
ConceptDebateReward: bridges the DebateEngine verdict into the rlgym reward system.

The reward shaping signal is computed asynchronously by the DebateEngine
(see train.py) and stored in a shared dict.  This reward function reads
the pre-computed shaping values for the current episode step.

Because the debate runs on trajectory segments (not single steps), the
shaping value is broadcast across the segment length — one debate verdict
contributes to cfg.DEBATE_SEQ_LEN timesteps.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

import config as cfg


class ConceptDebateReward:
    """
    Lightweight reward function that reads pre-computed debate shaping values.

    The training loop populates `self._pending_rewards` via `push_shaping()`.
    Each call to `get_rewards()` drains the queue for the current agents.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # agent_id -> list of pending shaping values
        self._pending_rewards: Dict[Any, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    def push_shaping(self, agent_id: Any, value: float) -> None:
        """Called by the training loop after each DebateEngine batch."""
        with self._lock:
            self._pending_rewards[agent_id].append(value)

    # ------------------------------------------------------------------
    def reset(self, agents: List, initial_state, shared_info: dict) -> None:
        with self._lock:
            for agent in agents:
                self._pending_rewards[agent].clear()

    def get_rewards(
        self,
        agents: List,
        state,
        is_terminated: Dict,
        is_truncated: Dict,
        shared_info: dict,
    ) -> Dict[Any, float]:
        rewards: Dict[Any, float] = {}
        with self._lock:
            for agent in agents:
                pending = self._pending_rewards[agent]
                if pending:
                    rewards[agent] = pending.pop(0)
                else:
                    rewards[agent] = 0.0
        return rewards
