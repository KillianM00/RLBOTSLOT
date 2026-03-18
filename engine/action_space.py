"""
90-action lookup table for Rocket League discrete action space.
Each action is an 8-dim vector:
  [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
  throttle/steer/pitch/yaw/roll in {-1, 0, 1}
  jump/boost/handbrake in {0, 1}

The 90 actions are the subset used by most competitive bots, matching
the rlgym LookupTableAction default table.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree  # type: ignore


def _build_lookup_table() -> np.ndarray:
    """Construct the canonical 90-action lookup table, shape (90, 8)."""
    actions = []

    # Ground actions: no jump/aerial controls
    for throttle in (-1.0, 0.0, 1.0):
        for steer in (-1.0, 0.0, 1.0):
            for boost in (0.0, 1.0):
                for handbrake in (0.0, 1.0):
                    # No boost + no throttle = idle (skip only one duplicate)
                    if boost == 1.0 and throttle == 0.0:
                        continue
                    actions.append([throttle, steer, 0.0, steer, 0.0,
                                    0.0, boost, handbrake])

    # Aerial actions: pitch, yaw, roll varied with boost/jump
    for pitch in (-1.0, 0.0, 1.0):
        for yaw in (-1.0, 0.0, 1.0):
            for roll in (-1.0, 0.0, 1.0):
                for jump in (0.0, 1.0):
                    for boost in (0.0, 1.0):
                        if pitch == 0.0 and yaw == 0.0 and roll == 0.0 and jump == 0.0:
                            continue  # duplicate of ground idle
                        actions.append([1.0, yaw, pitch, yaw, roll,
                                        jump, boost, 0.0])

    # Trim / pad to exactly 90
    table = np.array(actions[:90], dtype=np.float32)
    return table


# Module-level constant — built once, shared across workers via copy-on-fork
LOOKUP_TABLE: np.ndarray = _build_lookup_table()
N_ACTIONS: int = len(LOOKUP_TABLE)  # 90


class ActionTable:
    """Utility wrapper around the lookup table."""

    TABLE: np.ndarray = LOOKUP_TABLE

    @classmethod
    def get_action(cls, idx: int) -> np.ndarray:
        return cls.TABLE[idx]

    @classmethod
    def nearest_index(cls, raw_action: np.ndarray) -> int:
        """
        Find the lookup-table index closest to a continuous 8-dim action vector.
        Uses cosine distance via KD-tree on normalised vectors.
        Builds the tree lazily on first call (cached as class attribute).
        """
        if not hasattr(cls, "_kdtree"):
            norms = np.linalg.norm(cls.TABLE, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            cls._normed = cls.TABLE / norms
            cls._kdtree = KDTree(cls._normed)

        raw = np.asarray(raw_action, dtype=np.float32)
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm
        _, idx = cls._kdtree.query(raw)
        return int(idx)
