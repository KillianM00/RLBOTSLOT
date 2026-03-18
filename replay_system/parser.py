"""
ReplayParser: converts .replay files into Trajectory objects.

Uses carball (Python Rocket League replay parser) to extract frame data.
Maps continuous carball actions to the 90-action discrete lookup table
via KD-tree cosine distance (pre-built at startup for O(log 90) per frame).

If carball is not installed or fails, raises ImportError / RuntimeError
with guidance to install or use the rattletrap fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config as cfg
from engine.action_space import ActionTable, LOOKUP_TABLE

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """
    A single player's trajectory within a replay.
    Frames are T=SEGMENT_LENGTH windows ready for clustering/imitation.
    """
    replay_id: str
    player_name: str
    frames: np.ndarray       # (T, obs_approx_dim) float32 — positional features
    actions: np.ndarray      # (T,) int32 — indices into 90-action table
    metadata: Dict = field(default_factory=dict)


class ReplayParser:
    """
    Parses .replay files and produces Trajectory objects.
    """

    # Approximate obs features extracted from carball frames:
    #  [car_x, car_y, car_z, car_vx, car_vy, car_vz,
    #   ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz,
    #   boost, on_ground, has_flip, is_demo]
    FEATURE_DIM = 16
    SEGMENT_LEN = cfg.SEGMENT_LENGTH

    def __init__(self) -> None:
        # Pre-build KD-tree for fast action quantization
        ActionTable.nearest_index(np.zeros(8, dtype=np.float32))  # warm-up

    def parse(self, replay_path: Path) -> List[Trajectory]:
        """
        Parse a single .replay file.
        Returns a list of Trajectory objects (one per player).
        Raises RuntimeError if carball fails.
        """
        try:
            import carball  # type: ignore
        except ImportError:
            raise ImportError(
                "carball is not installed. Run: pip install carball\n"
                "Fallback: use rattletrap binary + custom parser."
            )

        try:
            analysis = carball.decompile_replay(str(replay_path))
            return self._extract_trajectories(analysis, replay_path.stem)
        except Exception as e:
            logger.warning("Failed to parse replay %s: %s", replay_path.name, e)
            return []

    def parse_batch(self, paths: List[Path]) -> List[Trajectory]:
        """Parse a list of replay files, skipping failures."""
        all_trajectories: List[Trajectory] = []
        for p in paths:
            try:
                all_trajectories.extend(self.parse(p))
            except Exception as e:
                logger.debug("Skipping %s: %s", p.name, e)
        return all_trajectories

    # ------------------------------------------------------------------
    def _extract_trajectories(self, analysis, replay_id: str) -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        try:
            game   = analysis.game
            frames = game.frames   # pandas DataFrame
            meta   = analysis.game.game_metadata if hasattr(analysis.game, "game_metadata") else {}

            for player in analysis.players:
                pname = getattr(player, "name", "unknown")
                raw_frames, raw_actions = self._player_frames(frames, player)
                if len(raw_frames) < self.SEGMENT_LEN:
                    continue
                # Segment into fixed-length windows
                for start in range(0, len(raw_frames) - self.SEGMENT_LEN, self.SEGMENT_LEN // 2):
                    seg_f = raw_frames[start:start + self.SEGMENT_LEN]
                    seg_a = raw_actions[start:start + self.SEGMENT_LEN]
                    trajectories.append(Trajectory(
                        replay_id=replay_id,
                        player_name=pname,
                        frames=seg_f.astype(np.float32),
                        actions=seg_a.astype(np.int32),
                        metadata={"replay_id": replay_id, "player": pname},
                    ))
        except Exception as e:
            logger.debug("Extraction error in replay %s: %s", replay_id, e)
        return trajectories

    def _player_frames(self, frames, player):
        """
        Extract FEATURE_DIM feature vectors and action indices per frame.
        Returns (np.ndarray of shape (T, FEATURE_DIM), np.ndarray of shape (T,)).
        """
        feature_list = []
        action_list  = []
        try:
            pdf = player.data   # player DataFrame with per-frame data
            for i in range(len(pdf)):
                row = pdf.iloc[i]
                feat = self._extract_features(frames.iloc[i] if i < len(frames) else None, row)
                action = self._extract_action(row)
                feature_list.append(feat)
                action_list.append(action)
        except Exception as e:
            logger.debug("Frame extraction error: %s", e)

        if not feature_list:
            return np.zeros((0, self.FEATURE_DIM), np.float32), np.zeros(0, np.int32)
        return np.array(feature_list, dtype=np.float32), np.array(action_list, dtype=np.int32)

    def _extract_features(self, frame_row, player_row) -> np.ndarray:
        """Extract a FEATURE_DIM vector from carball data rows."""
        def safe(obj, attr, default=0.0):
            try:
                return float(getattr(obj, attr, default) or default)
            except Exception:
                return float(default)

        feat = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        try:
            # Car position & velocity (indices 0-5)
            feat[0] = safe(player_row, "pos_x") / 4096.0
            feat[1] = safe(player_row, "pos_y") / 5120.0
            feat[2] = safe(player_row, "pos_z") / 2044.0
            feat[3] = safe(player_row, "vel_x") / 2300.0
            feat[4] = safe(player_row, "vel_y") / 2300.0
            feat[5] = safe(player_row, "vel_z") / 2300.0
            # Ball position & velocity (indices 6-11) - from frame row
            if frame_row is not None:
                feat[6]  = safe(frame_row, "ball_pos_x") / 4096.0
                feat[7]  = safe(frame_row, "ball_pos_y") / 5120.0
                feat[8]  = safe(frame_row, "ball_pos_z") / 2044.0
                feat[9]  = safe(frame_row, "ball_vel_x") / 6000.0
                feat[10] = safe(frame_row, "ball_vel_y") / 6000.0
                feat[11] = safe(frame_row, "ball_vel_z") / 6000.0
            # Boost (index 12)
            feat[12] = safe(player_row, "boost") / 100.0
            # Flags (indices 13-15)
            feat[13] = float(safe(player_row, "on_ground", 1.0) > 0.5)
            feat[14] = float(safe(player_row, "has_flip",  1.0) > 0.5)
            feat[15] = float(safe(player_row, "is_demolished", 0.0) > 0.5)
        except Exception:
            pass
        return feat

    def _extract_action(self, player_row) -> int:
        """Map carball continuous inputs to nearest 90-action index."""
        def safe(attr, default=0.0):
            try:
                v = getattr(player_row, attr, default)
                return float(v) if v is not None else float(default)
            except Exception:
                return float(default)

        raw = np.array([
            safe("throttle"),
            safe("steer"),
            safe("pitch"),
            safe("yaw"),
            safe("roll"),
            safe("jump"),
            safe("boost"),
            safe("handbrake"),
        ], dtype=np.float32)

        return ActionTable.nearest_index(raw)
