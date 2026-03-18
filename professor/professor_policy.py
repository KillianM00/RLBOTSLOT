"""
ProfessorPolicy: the full Mixture-of-Experts actor network.

This is the bot's "brain" — it routes each observation through the
GatingNetwork to select the most appropriate ExpertNetworks,
combines their outputs via weighted sum, and samples an action.

The "professor" metaphor in full:
  GatingNetwork  = Dean who decides which professors teach this lesson
  ExpertNetworks = Individual professors, each specialised in one concept
  DebateEngine   = Faculty review that evaluates teaching quality
  ConceptLibrary = The university catalogue of all courses offered

Interface: compatible with rlgym-ppo's expected actor API.
  get_action_and_value(obs) -> (action, log_prob, entropy, value)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import config as cfg
from professor.concept_library import ConceptLibrary
from professor.gating_network import GatingNetwork
from professor.expert_network import ExpertNetwork
from professor.debate_engine import DebateEngine, DebateVerdict
from models.critic_network import ValueNetwork


class ProfessorPolicy(nn.Module):
    """
    Full MoE actor.  Drop-in for rlgym-ppo's policy parameter.

    Forward pass:
      1. GatingNetwork selects top-k experts and returns gate weights
      2. Selected ExpertNetworks produce action logits
      3. Logits weighted-summed via gate_weights
      4. Categorical distribution sampled for the discrete action

    Also returns auxiliary losses for the training update:
      aux_losses = {
          "load_balance": lb_loss * LOAD_BALANCE_COEF,
          "debate":       debate_loss * DEBATE_LOSS_COEF,
      }
    """

    def __init__(
        self,
        concept_library: ConceptLibrary,
        obs_dim: int      = cfg.OBS_DIM,
        n_actions: int    = cfg.N_ACTIONS,
        max_concepts: int = cfg.MAX_N_CONCEPTS,
        k: int            = cfg.K_EXPERTS,
    ) -> None:
        super().__init__()
        self.concept_library = concept_library
        self.n_actions        = n_actions
        self.max_concepts     = max_concepts
        self.k                = k

        # GatingNetwork (manages active_mask internally)
        self.gating = GatingNetwork(
            obs_dim=obs_dim,
            max_concepts=max_concepts,
            k=k,
        )

        # Pre-allocated expert pool: MAX_N_CONCEPTS experts, most initially inactive
        self.experts = nn.ModuleList([
            ExpertNetwork(obs_dim=obs_dim, n_actions=n_actions)
            for _ in range(max_concepts)
        ])

        # Separate critic (not part of MoE to keep value estimate stable)
        self.critic = ValueNetwork(obs_dim=obs_dim)

        # Debate engine (asynchronous; called from training loop, not here)
        self.debate_engine = DebateEngine(
            obs_dim=obs_dim,
            n_actions=n_actions,
        )

        # Sync active_mask from ConceptLibrary into GatingNetwork at init
        self._sync_active_mask()

    # ------------------------------------------------------------------
    # Mask synchronisation
    # ------------------------------------------------------------------

    def _sync_active_mask(self) -> None:
        mask = torch.tensor(
            self.concept_library.active_mask,
            dtype=torch.bool,
        )
        self.gating.set_active_mask(mask)

    def notify_concept_update(self) -> None:
        """Call this after any ConceptLibrary.register() or prune()."""
        self._sync_active_mask()

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        obs : (batch, obs_dim)

        Returns
        -------
        action_logits : (batch, n_actions)
        aux_losses    : dict of named scalar tensors
        """
        gate_weights, top_k_indices, lb_loss = self.gating(obs)  # (B, max_c), (B, k)

        batch_size = obs.shape[0]

        # Collect logits from all active experts, zero for inactive
        # For efficiency: only call the top-k experts per sample
        combined_logits = torch.zeros(batch_size, self.n_actions, device=obs.device)

        # Gather gate weights for the top-k indices
        for ki in range(self.k):
            expert_indices = top_k_indices[:, ki]  # (B,)
            weights_for_k  = gate_weights.gather(1, top_k_indices[:, ki:ki+1])  # (B, 1)

            # Run each unique expert once and scatter results
            for eidx in expert_indices.unique():
                sample_mask = (expert_indices == eidx)  # bool (B,)
                if not sample_mask.any():
                    continue
                obs_subset = obs[sample_mask]
                logits_subset = self.experts[eidx.item()](obs_subset)  # (n_sub, n_act)
                w = weights_for_k[sample_mask]                          # (n_sub, 1)
                combined_logits[sample_mask] = combined_logits[sample_mask] + logits_subset * w

        aux_losses = {
            "load_balance": lb_loss * cfg.LOAD_BALANCE_COEF,
        }

        return combined_logits, aux_losses

    # ------------------------------------------------------------------
    # rlgym-ppo compatible interface
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs    : (batch, obs_dim)
        action : (batch,) long — if provided, evaluate log_prob of these actions
                                  (used during PPO update phase)

        Returns
        -------
        action    : (batch,) sampled or provided action indices
        log_prob  : (batch,) log probability of the action
        entropy   : scalar mean entropy
        value     : (batch,) estimated state values
        """
        logits, _ = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy  = dist.entropy().mean()
        value    = self.critic.get_value(obs)

        return action, log_prob, entropy, value

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference-only; no gradient tracking needed."""
        with torch.no_grad():
            logits, _ = self.forward(obs)
        return Categorical(logits=logits).sample()

    # ------------------------------------------------------------------
    # Active expert identification (for debate / logging)
    # ------------------------------------------------------------------

    def get_active_concept_ids(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the top-1 expert index and its gate weight for each sample.
        Used by training loop to label trajectory segments for debate.
        """
        with torch.no_grad():
            gate_weights, top_k_indices, _ = self.gating(obs)
        top1_idx    = top_k_indices[:, 0]           # (B,)
        top1_weight = gate_weights.gather(1, top_k_indices[:, :1]).squeeze(-1)  # (B,)
        return top1_idx, top1_weight
