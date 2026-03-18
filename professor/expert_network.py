"""
ExpertNetwork: per-concept policy head.

One ExpertNetwork per concept slot.  They share no parameters
(each is an independent specialist), but they are instantiated inside
ProfessorPolicy's nn.ModuleList for batched gradient updates.

Architecture: obs (obs_dim) -> [512, 256] -> action_logits (90)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import config as cfg
from models.base_policy import build_mlp


class ExpertNetwork(nn.Module):
    """
    Single-expert policy head.

    The "professor" metaphor: each ExpertNetwork is a professor who
    specialises in one concept.  It produces action logits when the
    GatingNetwork routes observations to it.
    """

    def __init__(
        self,
        obs_dim: int        = cfg.OBS_DIM,
        n_actions: int      = cfg.N_ACTIONS,
        hidden_sizes: Tuple = cfg.EXPERT_HIDDEN,
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (batch, obs_dim)

        Returns
        -------
        action_logits : (batch, n_actions)
        """
        return self.net(obs)
