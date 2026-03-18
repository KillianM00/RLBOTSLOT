"""
AsyncReplayFetcher: wraps BallchasingFetcher in a background daemon thread.

Replay fetching is IO-bound and slow; we never want it to block the
main training loop.  Instead, we:
  1. Start a daemon thread that runs a fetch every REPLAY_FETCH_INTERVAL steps
  2. Results are placed on a Queue for the training loop to consume
  3. The training loop checks the queue at convenient moments (between PPO iters)

Usage:
    fetcher = AsyncReplayFetcher(ballchasing_fetcher, parser, curriculum)
    fetcher.start()                       # launches background thread
    ...
    new_subjects = fetcher.drain()        # non-blocking, returns list
    ...
    fetcher.stop()
"""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import List, Optional

import config as cfg

logger = logging.getLogger(__name__)


class AsyncReplayFetcher:
    """
    Background thread that periodically fetches + parses replays
    and delivers discovered subjects to the main training loop.
    """

    def __init__(
        self,
        ballchasing_fetcher,    # BallchasingFetcher instance
        replay_parser,          # ReplayParser instance
        behavior_clusterer,     # BehaviorClusterer instance
        subject_namer,          # SubjectNamer instance
        subject_curriculum,     # SubjectCurriculum instance
        fetch_interval: int = cfg.REPLAY_FETCH_INTERVAL,
    ) -> None:
        self.fetcher           = ballchasing_fetcher
        self.parser            = replay_parser
        self.clusterer         = behavior_clusterer
        self.namer             = subject_namer
        self.curriculum        = subject_curriculum
        self.fetch_interval    = fetch_interval

        self._result_queue: queue.Queue = queue.Queue()
        self._stop_event   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._timesteps    = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("AsyncReplayFetcher started (interval=%d steps)", self.fetch_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def notify_timesteps(self, n: int) -> None:
        """Called by the training loop each PPO iteration."""
        self._timesteps += n

    def drain(self) -> list:
        """Non-blocking: drain all pending results from the queue."""
        results = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    # ------------------------------------------------------------------
    def _run(self) -> None:
        """Background thread main loop."""
        last_fetch_ts = 0
        while not self._stop_event.is_set():
            if self._timesteps - last_fetch_ts >= self.fetch_interval:
                last_fetch_ts = self._timesteps
                self._do_fetch_cycle()
            self._stop_event.wait(timeout=30.0)   # check every 30s

    def _do_fetch_cycle(self) -> None:
        """One complete fetch → parse → cluster → name → register cycle."""
        try:
            logger.info("AsyncReplayFetcher: starting fetch cycle at step %d",
                        self._timesteps)

            # 1. Download replays
            paths = self.fetcher.fetch_batch_sync()
            if not paths:
                return

            # 2. Parse into trajectories
            trajectories = self.parser.parse_batch(paths)
            if not trajectories:
                return
            logger.info("Parsed %d trajectory segments", len(trajectories))

            # 3. Cluster behaviours
            labels        = self.clusterer.fit(trajectories)
            feature_matrix = None  # will be computed inside if needed
            exemplars     = self.clusterer.get_cluster_exemplars(labels, trajectories)

            # 4. Name clusters
            unique_ids = [c for c in set(labels) if c != -1]
            if not unique_ids:
                return
            import numpy as np
            feature_matrix = np.stack([
                self.clusterer.extract_features(t) for t in trajectories
            ])
            cluster_trajs = {
                cid: [t for t, l in zip(trajectories, labels) if l == cid]
                for cid in unique_ids
            }
            subject_names = self.namer.name_all_clusters(
                np.array(unique_ids), exemplars, feature_matrix, labels
            )

            # 5. Register with curriculum
            registered = self.curriculum.register_subjects(
                subject_names, exemplars, cluster_trajs
            )
            if registered:
                logger.info("Discovered %d new subjects: %s",
                            len(registered), [r[0] for r in registered])
                self._result_queue.put(registered)

        except Exception as e:
            logger.error("AsyncReplayFetcher fetch cycle failed: %s", e, exc_info=True)
