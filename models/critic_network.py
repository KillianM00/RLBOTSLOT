"""
Centralised value network (critic) used by PPO.
Kept deliberately simple and separate from the MoE actor to avoid
debate/gating losses corrupting the value estimate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import config as cfg
from models.base_policy import build_mlp


class ValueNetwork(nn.Module):
    """
    Standard MLP critic: obs -> scalar value estimate V(s).

    Architecture: [obs_dim] -> [512, 256, 128] -> [1]
    """

    def __init__(self, obs_dim: int = cfg.OBS_DIM) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, (512, 256, 128), 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (batch, obs_dim)

        Returns
        -------
        value : (batch, 1)
        """
        return self.net(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Convenience alias matching rlgym-ppo's expected interface."""
        return self.forward(obs).squeeze(-1)
