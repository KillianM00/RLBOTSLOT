"""
ImitationLearner: behavioural cloning (BC) pre-trainer.

Given a list of Trajectory objects associated with a specific named subject,
trains the corresponding ExpertNetwork to imitate the demonstrated actions
using cross-entropy loss.

After BC pre-training, the expert is handed off to RL fine-tuning
(handled in train.py) where a small BC regularisation loss is mixed
in for the first BC_REGULARIZE_STEPS timesteps.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from professor.expert_network import ExpertNetwork
from replay_system.parser import Trajectory

logger = logging.getLogger(__name__)


class ImitationLearner:
    """
    Trains an ExpertNetwork via behavioural cloning on a set of trajectories.
    """

    def __init__(
        self,
        expert: ExpertNetwork,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.expert    = expert.to(device)
        self.device    = device
        self.optimizer = optim.Adam(expert.parameters(), lr=3e-4, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def train(
        self,
        trajectories: List[Trajectory],
        n_epochs: int  = cfg.BC_EPOCHS,
        batch_size: int = cfg.BC_BATCH_SIZE,
    ) -> float:
        """
        Run BC training.  Returns final epoch cross-entropy loss.
        """
        if not trajectories:
            return 0.0

        # Build dataset: concatenate all trajectory frames and actions
        all_obs     = np.concatenate([t.frames   for t in trajectories], axis=0)
        all_actions = np.concatenate([t.actions  for t in trajectories], axis=0)

        # Pad / trim obs to match ExpertNetwork's obs_dim
        obs_dim    = cfg.OBS_DIM
        feat_dim   = all_obs.shape[1]
        if feat_dim < obs_dim:
            pad = np.zeros((all_obs.shape[0], obs_dim - feat_dim), dtype=np.float32)
            all_obs = np.concatenate([all_obs, pad], axis=1)
        elif feat_dim > obs_dim:
            all_obs = all_obs[:, :obs_dim]

        obs_tensor = torch.tensor(all_obs,    dtype=torch.float32)
        act_tensor = torch.tensor(all_actions, dtype=torch.long)
        dataset    = torch.utils.data.TensorDataset(obs_tensor, act_tensor)
        loader     = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        self.expert.train()
        final_loss = 0.0

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for obs_b, act_b in loader:
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)

                logits = self.expert(obs_b)
                loss   = self.criterion(logits, act_b)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.expert.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg = epoch_loss / max(len(loader), 1)
            logger.debug("BC epoch %d/%d | loss=%.4f", epoch + 1, n_epochs, avg)
            final_loss = avg

        self.expert.eval()
        logger.info("BC pre-training complete | final_loss=%.4f | n_samples=%d",
                    final_loss, len(all_obs))
        return final_loss

    # ------------------------------------------------------------------
    def bc_regularization_loss(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Called during RL fine-tuning for the first BC_REGULARIZE_STEPS timesteps.
        Returns cross-entropy loss to mix with PPO loss.
        """
        logits = self.expert(obs_batch)
        return self.criterion(logits, action_batch)
