"""
SubjectNamer: assigns human-readable names to discovered behavior clusters.

Names are derived heuristically from the dominant feature characteristics
of each cluster.  A unique suffix (cluster_id) prevents collisions.

The "subject" metaphor: each cluster becomes a new course in the professor
curriculum, e.g. "AerialRecovery_3", "FlickSequence_7", "CeilingShot_4".
"""
from __future__ import annotations

from typing import Dict, Set

import numpy as np

import config as cfg
from replay_system.parser import Trajectory


# Feature indices within BehaviorClusterer.extract_features() output
_F_MEAN_BALL_Z   = 0
_F_MAX_BALL_Z    = 1
_F_MEAN_CAR_Z    = 2
_F_AERIAL_FRAC   = 3
_F_MAX_DELTA_VEL = 4
_F_MEAN_BALL_SPD = 5
_F_BOOST_USED    = 6
_F_MEAN_BOOST    = 7
_F_Y_PROGRESS    = 8
_F_GROUND_FRAC   = 9
_F_MEAN_CAR_SPD  = 10
_F_TOUCH_COUNT   = 11


def _infer_prefix(feat: np.ndarray) -> str:
    """
    Heuristically pick a subject prefix from SUBJECT_PREFIXES
    based on the dominant feature values of the cluster.
    """
    aerial_frac   = feat[_F_AERIAL_FRAC]
    max_ball_z    = feat[_F_MAX_BALL_Z]    # 0-1 normalised
    max_delta_vel = feat[_F_MAX_DELTA_VEL]
    y_progress    = feat[_F_Y_PROGRESS]
    ground_frac   = feat[_F_GROUND_FRAC]
    boost_used    = feat[_F_BOOST_USED]
    touch_count   = feat[_F_TOUCH_COUNT]
    mean_boost    = feat[_F_MEAN_BOOST]
    mean_ball_z   = feat[_F_MEAN_BALL_Z]

    # Decision tree of feature dominance
    if aerial_frac > 0.7 and max_ball_z > 0.4:
        return "CeilingShot"
    if aerial_frac > 0.5:
        if max_delta_vel > 0.6:
            return "AerialRecovery"
        return "AerialRecovery"
    if max_delta_vel > 0.7 and ground_frac > 0.5 and mean_ball_z < 0.15:
        return "FlickSequence"
    if max_delta_vel > 0.5 and y_progress > 0.2:
        return "RollingGroundShot"
    if boost_used < 0.05 and ground_frac > 0.8:
        return "ShadowDefense"
    if boost_used > 0.4 and y_progress > 0.15:
        return "BoostDeny"
    if mean_boost < 0.15 and boost_used < 0.1:
        return "BoostStarve"
    if touch_count > 0.05 and aerial_frac > 0.3:
        return "ResetChain"
    if y_progress < -0.1 and ground_frac > 0.8:
        return "RotationCover"
    if max_delta_vel > 0.5 and touch_count > 0.03:
        return "DoubleTap"
    if ground_frac > 0.7 and max_delta_vel > 0.4:
        return "MustyFlick"
    if y_progress < 0.0 and aerial_frac < 0.2:
        return "NeutralRead"
    return "DirectionChange"


class SubjectNamer:
    """
    Assigns deterministic human-readable names to behavior clusters.
    """

    def __init__(self) -> None:
        self._used_names: Set[str] = set()

    def name_cluster(
        self,
        cluster_id: int,
        exemplar: Trajectory,
        cluster_features: np.ndarray,  # mean feature vector of the cluster
    ) -> str:
        """
        Returns a unique subject name like "AerialRecovery_3".
        """
        prefix = _infer_prefix(cluster_features)
        name   = f"{prefix}_{cluster_id}"
        # Ensure uniqueness
        counter = 0
        base    = name
        while name in self._used_names:
            counter += 1
            name = f"{base}_{counter}"
        self._used_names.add(name)
        return name

    def name_all_clusters(
        self,
        cluster_ids: np.ndarray,    # unique cluster ids (excluding -1)
        exemplars: Dict[int, Trajectory],
        feature_matrix: np.ndarray,  # (n_trajectories, feat_dim)
        labels: np.ndarray,           # (n_trajectories,) cluster labels
    ) -> Dict[int, str]:
        """
        Returns {cluster_id: subject_name} for all valid clusters.
        """
        names: Dict[int, str] = {}
        for cid in cluster_ids:
            mask = labels == cid
            mean_feat = feature_matrix[mask].mean(axis=0)
            exemplar  = exemplars.get(int(cid))
            if exemplar is None:
                continue
            names[int(cid)] = self.name_cluster(int(cid), exemplar, mean_feat)
        return names
