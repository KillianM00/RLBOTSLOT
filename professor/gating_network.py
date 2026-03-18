"""
GatingNetwork: routes each observation to the top-k most relevant ExpertNetworks.

Implements noisy top-k gating from "Outrageously Large Neural Networks"
(Shazeer et al. 2017) with an auxiliary load-balance loss to prevent
all traffic routing to a single expert.

Design choice: pre-allocate MAX_N_CONCEPTS=64 expert slots and use
`active_mask` to hide inactive experts at routing time.
New concepts are activated by flipping their slot in the mask,
with no architecture change required.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.base_policy import build_mlp


class GatingNetwork(nn.Module):
    """
    Soft noisy top-k router.

    Forward returns:
      gate_weights  : (batch, MAX_N_CONCEPTS) — non-zero only for top-k slots
      top_k_indices : (batch, k)              — indices of selected experts
      lb_loss       : scalar                  — auxiliary load-balance loss
    """

    def __init__(
        self,
        obs_dim: int       = cfg.OBS_DIM,
        max_concepts: int  = cfg.MAX_N_CONCEPTS,
        k: int             = cfg.K_EXPERTS,
        hidden_sizes: Tuple= cfg.GATING_HIDDEN,
        noise_std: float   = cfg.GATING_NOISE_STD,
    ) -> None:
        super().__init__()
        self.max_concepts  = max_concepts
        self.k             = k
        self.noise_std     = noise_std

        # Trunk: compress observation into a routing context vector
        self.trunk = build_mlp(obs_dim, hidden_sizes, hidden_sizes[-1])

        # Gating logits (W_g) and noise weights (W_n) in Shazeer notation
        self.W_gate  = nn.Linear(hidden_sizes[-1], max_concepts, bias=False)
        self.W_noise = nn.Linear(hidden_sizes[-1], max_concepts, bias=False)

        # Active-slot mask: shape (max_concepts,) bool; managed externally
        self.register_buffer(
            "active_mask",
            torch.zeros(max_concepts, dtype=torch.bool),
        )
        # Activate initial concept slots
        self.active_mask[:cfg.INITIAL_N_CONCEPTS] = True

    # ------------------------------------------------------------------
    def set_active_mask(self, mask: torch.Tensor) -> None:
        """Called when ConceptLibrary adds/prunes concepts."""
        self.active_mask.copy_(mask.to(self.active_mask.device))

    # ------------------------------------------------------------------
    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : (batch, obs_dim)

        Returns
        -------
        gate_weights  : (batch, max_concepts)  — sparse (0 for non-top-k)
        top_k_indices : (batch, k)
        lb_loss       : scalar tensor
        """
        ctx = self.trunk(obs)                          # (batch, hidden)
        logits = self.W_gate(ctx)                      # (batch, max_concepts)

        # Add multiplicative noise during training
        if self.training:
            noise_w   = F.softplus(self.W_noise(ctx))  # (batch, max_concepts)
            noise     = torch.randn_like(logits) * noise_w * self.noise_std
            logits    = logits + noise

        # Mask out inactive slots with a large negative value so they never win
        # (avoid float("-inf") which can cause NaN in softmax on some backends)
        inactive = ~self.active_mask.unsqueeze(0).expand_as(logits)
        logits   = logits - inactive.float() * 1e9

        # Top-k selection
        top_k_logits, top_k_indices = logits.topk(self.k, dim=-1)  # (B, k)

        # Sparse gate weights: softmax over top-k, zero elsewhere
        top_k_weights = F.softmax(top_k_logits, dim=-1)            # (B, k)

        # Build gate_weights without any scatter/one_hot (DirectML-compatible):
        # Compare each expert index against the full range via broadcasting.
        #   top_k_indices : (B, k, 1)
        #   expert_range  : (1, 1, max_c)
        #   mask          : (B, k, max_c)  — 1.0 where expert matches
        #   gate_weights  : (B, max_c) = sum over k of mask * weight
        expert_range = torch.arange(
            self.max_concepts, device=logits.device
        ).view(1, 1, self.max_concepts)                              # (1, 1, max_c)
        idx_expanded = top_k_indices.unsqueeze(-1)                  # (B, k, 1)
        mask         = (idx_expanded == expert_range).float()        # (B, k, max_c)
        gate_weights = (mask * top_k_weights.unsqueeze(-1)).sum(1)  # (B, max_c)

        # Auxiliary load-balance loss (encourages uniform expert utilisation)
        # fraction of tokens routed to each expert
        importance = gate_weights.sum(0)                            # (max_c,)
        # only over active experts
        n_active   = self.active_mask.sum().float().clamp(min=1)
        mean_imp   = importance.sum() / n_active
        # coefficient-of-variation squared
        active_imp = importance[self.active_mask]
        lb_loss    = (active_imp.std() / (active_imp.mean() + 1e-6)) ** 2

        return gate_weights, top_k_indices, lb_loss
