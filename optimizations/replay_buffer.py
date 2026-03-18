"""
MemoryMappedReplayBuffer: off-heap experience storage using numpy.memmap.

All tensors stored as float16 obs (halving memory vs float32).
Process safety (multiple writers from worker processes):
  - Each worker writes to a pre-assigned exclusive index range.
  - The shared write pointer is managed via multiprocessing.Value + Lock.
  - For single-process use, a simple int suffices.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import config as cfg

logger = logging.getLogger(__name__)


class MemoryMappedReplayBuffer:
    """
    Ring-buffer backed by numpy.memmap files on disk.

    Storage layout (per timestep):
      obs       : (obs_dim,)  float16   — current observation
      next_obs  : (obs_dim,)  float16   — next observation
      action    : ()          uint8     — action index (0-89)
      reward    : ()          float32
      done      : ()          bool

    Files are created lazily on first write.
    """

    def __init__(
        self,
        capacity: int  = cfg.REPLAY_BUFFER_CAPACITY,
        obs_dim: int   = cfg.OBS_DIM,
        path: Path     = cfg.MEMMAP_PATH,
    ) -> None:
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.path     = Path(path)

        # Shared write pointer (process-safe)
        self._ptr  = mp.Value("i", 0)
        self._size = mp.Value("i", 0)
        self._lock = mp.Lock()

        # Memory-mapped arrays (opened lazily)
        self._obs:      Optional[np.memmap] = None
        self._next_obs: Optional[np.memmap] = None
        self._action:   Optional[np.memmap] = None
        self._reward:   Optional[np.memmap] = None
        self._done:     Optional[np.memmap] = None
        self._init_memmaps()

    # ------------------------------------------------------------------
    def _init_memmaps(self) -> None:
        mode = "r+" if self.path.with_suffix(".obs.dat").exists() else "w+"
        kw   = dict(dtype=np.float16, shape=(self.capacity, self.obs_dim), mode=mode)

        self._obs      = np.memmap(str(self.path) + ".obs.dat",     **kw)
        self._next_obs = np.memmap(str(self.path) + ".nobs.dat",    **kw)
        self._action   = np.memmap(str(self.path) + ".act.dat",
                                   dtype=np.uint8,   shape=(self.capacity,), mode=mode)
        self._reward   = np.memmap(str(self.path) + ".rew.dat",
                                   dtype=np.float32, shape=(self.capacity,), mode=mode)
        self._done     = np.memmap(str(self.path) + ".done.dat",
                                   dtype=bool,       shape=(self.capacity,), mode=mode)
        logger.debug("MemoryMappedReplayBuffer initialised (capacity=%d, obs_dim=%d)",
                     self.capacity, self.obs_dim)

    # ------------------------------------------------------------------
    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        """Thread/process-safe push to the ring buffer."""
        with self._lock:
            idx = self._ptr.value
            self._obs[idx]      = obs.astype(np.float16)
            self._next_obs[idx] = next_obs.astype(np.float16)
            self._action[idx]   = np.uint8(action)
            self._reward[idx]   = np.float32(reward)
            self._done[idx]     = bool(done)
            self._ptr.value     = (idx + 1) % self.capacity
            if self._size.value < self.capacity:
                self._size.value += 1

    # ------------------------------------------------------------------
    def sample(self, n: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch from the buffer.
        Returns (obs, actions, rewards, dones, next_obs) as float32 tensors.
        """
        size = self._size.value
        if size == 0:
            raise RuntimeError("Replay buffer is empty.")
        idxs = np.random.randint(0, size, size=n)
        obs      = torch.tensor(self._obs[idxs].astype(np.float32))
        next_obs = torch.tensor(self._next_obs[idxs].astype(np.float32))
        actions  = torch.tensor(self._action[idxs].astype(np.int64))
        rewards  = torch.tensor(self._reward[idxs])
        dones    = torch.tensor(self._done[idxs].astype(np.float32))
        return obs, actions, rewards, dones, next_obs

    def __len__(self) -> int:
        return self._size.value
