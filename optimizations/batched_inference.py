"""
BatchedInference: collects observations from multiple worker processes,
runs a single GPU forward pass, and distributes actions back.

Reduces GPU kernel launch overhead from O(n_workers) to O(1) per step.
Uses threading.Barrier to synchronise workers before each batch inference.

Note: In rlgym-ppo's multiprocessing setup, the policy is typically called
in the learner process, not in workers.  This class is primarily useful
for custom rollout collection outside rlgym-ppo's standard loop, or for
profiling purposes.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import torch

import config as cfg


class BatchedInference:
    """
    Thread-safe batched forward pass for the ProfessorPolicy.

    Usage (from multiple worker threads):
        inference = BatchedInference(policy, n_workers=32)
        # Each worker calls:
        action = inference.infer_one(worker_id, obs)
    """

    def __init__(
        self,
        policy,                          # ProfessorPolicy
        n_workers: int  = cfg.N_WORKERS,
        device: str     = "cuda" if torch.cuda.is_available() else "cpu",
        min_batch: int  = cfg.MIN_INFERENCE_BATCH,
    ) -> None:
        self.policy     = policy
        self.n_workers  = n_workers
        self.device     = device
        self.min_batch  = min_batch

        # Shared buffers
        obs_dim = cfg.OBS_DIM
        self._obs_buf     = np.zeros((n_workers, obs_dim), dtype=np.float32)
        self._action_buf  = np.zeros(n_workers, dtype=np.int64)
        self._ready       = np.zeros(n_workers, dtype=bool)
        self._lock        = threading.Lock()
        self._results_ev  = [threading.Event() for _ in range(n_workers)]
        self._batch_ev    = threading.Event()
        self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._batch_thread.start()

    # ------------------------------------------------------------------
    def infer_one(self, worker_id: int, obs: np.ndarray) -> int:
        """
        Submit one observation and block until action is returned.
        Called from worker threads.
        """
        with self._lock:
            self._obs_buf[worker_id] = obs
            self._ready[worker_id]   = True
            self._results_ev[worker_id].clear()
            if self._ready.sum() >= self.min_batch:
                self._batch_ev.set()

        self._results_ev[worker_id].wait()
        return int(self._action_buf[worker_id])

    def infer_batch(self, obs_list: List[np.ndarray]) -> List[int]:
        """
        Synchronous batch inference (no threading).
        Used from the main process during evaluation.
        """
        obs_t = torch.tensor(
            np.stack(obs_list), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            actions = self.policy.get_action(obs_t)
        return actions.cpu().tolist()

    # ------------------------------------------------------------------
    def _batch_loop(self) -> None:
        """Background thread: waits for enough workers, then runs GPU inference."""
        while True:
            self._batch_ev.wait()
            self._batch_ev.clear()

            with self._lock:
                mask   = self._ready.copy()
                obs_in = self._obs_buf[mask].copy()
                idxs   = np.where(mask)[0]
                self._ready[:] = False

            if len(obs_in) == 0:
                continue

            obs_t = torch.tensor(obs_in, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions = self.policy.get_action(obs_t).cpu().numpy()

            for local_i, global_i in enumerate(idxs):
                self._action_buf[global_i] = actions[local_i]
                self._results_ev[global_i].set()
