"""
BehaviorClusterer: discovers behavior archetypes in replay trajectories.

Approach:
  1. Extract hand-crafted feature vectors from each Trajectory
     (FEATURE_DIM = 64, hand-crafted from physics features)
  2. Run HDBSCAN density-based clustering
  3. Return cluster labels (-1 = noise/outlier) and exemplar trajectories

Feature extraction focuses on:
  - Ball height (aerial vs ground play)
  - Car height
  - Delta ball velocity (large → flick/hit)
  - Boost usage rate
  - Touch frequency
  - Ball-to-goal alignment
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg
from replay_system.parser import Trajectory

logger = logging.getLogger(__name__)


class BehaviorClusterer:
    """
    Clusters trajectory segments into behavior archetypes using HDBSCAN.
    """

    FEATURE_DIM = 12   # per-segment feature vector length

    def __init__(
        self,
        min_cluster_size: int = cfg.HDBSCAN_MIN_CLUSTER,
        min_samples: int      = cfg.HDBSCAN_MIN_SAMPLES,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples      = min_samples
        self._clusterer = None

    # ------------------------------------------------------------------
    def extract_features(self, traj: Trajectory) -> np.ndarray:
        """
        Compute a FEATURE_DIM summary vector for a single trajectory segment.
        """
        frames = traj.frames    # (T, raw_feat_dim)
        T      = frames.shape[0]
        if T == 0:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # Indices in Trajectory.frames (see parser.py):
        #  0-2: car_pos (normalised)  3-5: car_vel  6-8: ball_pos  9-11: ball_vel
        # 12: boost  13: on_ground  14: has_flip  15: is_demo

        car_z    = frames[:, 2] * 2044.0   # un-normalise
        ball_z   = frames[:, 8] * 2044.0

        car_vel  = frames[:, 3:6]           # normalised
        ball_vel = frames[:, 9:12]

        boost    = frames[:, 12]
        on_ground = frames[:, 13]

        # 1. Mean ball height (aerial indicator)
        mean_ball_z  = float(np.mean(ball_z))

        # 2. Max ball height
        max_ball_z   = float(np.max(ball_z))

        # 3. Mean car height
        mean_car_z   = float(np.mean(car_z))

        # 4. Aerial fraction (car z > 100 uu)
        aerial_frac  = float(np.mean(car_z > 100.0))

        # 5. Ball delta velocity magnitude (flick / shot indicator)
        if T > 1:
            ball_vel_delta = np.diff(ball_vel * 6000.0, axis=0)  # un-normalise
            max_delta      = float(np.max(np.linalg.norm(ball_vel_delta, axis=1)))
        else:
            max_delta = 0.0

        # 6. Mean ball speed
        mean_ball_speed = float(np.mean(np.linalg.norm(ball_vel * 6000.0, axis=1)))

        # 7. Boost usage rate (fraction of time boost was active)
        if T > 1:
            boost_used  = float(np.mean(np.diff(boost) < -0.005))
        else:
            boost_used  = 0.0

        # 8. Mean boost level
        mean_boost   = float(np.mean(boost))

        # 9. Ball y-progression (toward orange goal = positive)
        ball_y_start = float(frames[0, 7]) * 5120.0
        ball_y_end   = float(frames[-1, 7]) * 5120.0
        y_progress   = (ball_y_end - ball_y_start) / 5120.0

        # 10. Ground fraction
        ground_frac  = float(np.mean(on_ground))

        # 11. Car speed mean
        mean_car_speed = float(np.mean(np.linalg.norm(car_vel * 2300.0, axis=1)))

        # 12. Number of touches (proxy: large ball velocity delta > threshold)
        touch_count   = float(np.sum(
            np.linalg.norm(np.diff(ball_vel * 6000.0, axis=0), axis=1) > 300.0
        )) / max(T, 1)

        feat = np.array([
            mean_ball_z / 2044.0,
            max_ball_z  / 2044.0,
            mean_car_z  / 2044.0,
            aerial_frac,
            max_delta   / 6000.0,
            mean_ball_speed / 6000.0,
            boost_used,
            mean_boost,
            y_progress,
            ground_frac,
            mean_car_speed / 2300.0,
            touch_count,
        ], dtype=np.float32)

        return feat

    # ------------------------------------------------------------------
    def fit(self, trajectories: List[Trajectory]) -> np.ndarray:
        """
        Fit HDBSCAN on extracted features.
        Returns cluster labels array of shape (len(trajectories),).
        -1 = noise / outlier.
        """
        if not trajectories:
            return np.array([], dtype=np.int32)

        feature_matrix = np.stack([self.extract_features(t) for t in trajectories])

        try:
            import hdbscan  # type: ignore
        except ImportError:
            logger.warning("hdbscan not installed; falling back to k-means (k=8).")
            return self._kmeans_fallback(feature_matrix)

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        labels = self._clusterer.fit_predict(feature_matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info("BehaviorClusterer: found %d clusters in %d trajectories",
                    n_clusters, len(trajectories))
        return labels.astype(np.int32)

    def _kmeans_fallback(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans  # type: ignore
        k = min(8, len(X))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        return km.fit_predict(X).astype(np.int32)

    # ------------------------------------------------------------------
    def get_cluster_exemplars(
        self,
        labels: np.ndarray,
        trajectories: List[Trajectory],
        feature_matrix: Optional[np.ndarray] = None,
    ) -> Dict[int, Trajectory]:
        """
        For each cluster, find the trajectory closest to the cluster centroid.
        Returns {cluster_id: exemplar_trajectory}.
        """
        if feature_matrix is None:
            feature_matrix = np.stack([self.extract_features(t) for t in trajectories])

        exemplars: Dict[int, Trajectory] = {}
        unique_labels = set(labels) - {-1}

        for cid in unique_labels:
            mask    = labels == cid
            cluster_feats = feature_matrix[mask]
            centroid = cluster_feats.mean(axis=0)
            dists    = np.linalg.norm(cluster_feats - centroid, axis=1)
            local_idx = int(np.argmin(dists))
            global_idx = np.where(mask)[0][local_idx]
            exemplars[int(cid)] = trajectories[global_idx]

        return exemplars
