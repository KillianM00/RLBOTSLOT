"""
DebateEngine: the core of the "professor argues with itself" mechanism.

The metaphor:
  Each ExpertNetwork is a professor teaching a concept (e.g. "AerialRecovery_3").
  After the bot executes an action in that concept's context, the DebateEngine
  runs two adversarial heads:

  Proponent ("This execution was correct"):
      P(correct | obs_sequence, action_sequence, concept_id)

  Opponent ("This was wrong because..."):
      P(incorrect | obs_sequence, action_sequence, concept_id)

  The debate score drives ConceptDebateReward: proponent wins → positive shaping;
  opponent wins → negative shaping, and the ExpertNetwork is penalised via
  debate_loss (discourages its outputs for that concept).

Architecture:
  - GRU trajectory encoder (seq_len=30 frames, hidden=256)
  - Concept embedding injected at every GRU step
  - Proponent head: linear -> sigmoid
  - Opponent head:  linear -> sigmoid
  Both share the GRU encoder but not the final head weights.

Run asynchronously (see optimizations/async_fetcher.py); capped at
MAX_DEBATES_PER_ITER debates per PPO iteration.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.embedding_network import ConceptEmbeddingNetwork


class DebateVerdict(NamedTuple):
    proponent_score: torch.Tensor   # (batch,) in [0,1]
    opponent_score: torch.Tensor    # (batch,) in [0,1]
    winner: List[str]               # "proponent" | "opponent" | "draw"
    shaping_reward: torch.Tensor    # (batch,) signed reward signal


class DebateEngine(nn.Module):
    """
    Processes trajectory segments and produces debate verdicts.

    Inputs (per debate):
      obs_seq    : (T, batch, obs_dim)   — observation sequence
      action_seq : (T, batch)            — action indices
      concept_ids: (batch,)              — which concept was active

    Output: DebateVerdict
    """

    def __init__(
        self,
        obs_dim: int         = cfg.OBS_DIM,
        n_actions: int       = cfg.N_ACTIONS,
        gru_hidden: int      = cfg.DEBATE_GRU_HIDDEN,
        embed_dim: int       = cfg.EMBED_DIM,
        max_concepts: int    = cfg.MAX_N_CONCEPTS,
    ) -> None:
        super().__init__()
        self.gru_hidden = gru_hidden

        # Concept embeddings (shared with ConceptLibrary via weight sync)
        self.concept_embed = ConceptEmbeddingNetwork(max_concepts, embed_dim)

        # Action embedding: map 90 discrete actions to a dense vector
        self.action_embed = nn.Embedding(n_actions, embed_dim)

        # GRU trajectory encoder
        # Input at each step: [obs, action_embed, concept_embed] concatenated
        self.gru = nn.GRU(
            input_size=obs_dim + embed_dim + embed_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=False,   # (T, B, input)
            dropout=0.1,
        )

        # Proponent and Opponent heads (independent final layers)
        self.proponent_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.opponent_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        obs_seq: torch.Tensor,      # (T, B, obs_dim)
        action_seq: torch.Tensor,   # (T, B) long
        concept_ids: torch.Tensor,  # (B,) long
    ) -> DebateVerdict:
        # DebateEngine always runs on CPU — GRU is not fully supported on DirectML.
        # Inputs are moved to CPU here; results are returned as CPU tensors.
        cpu = torch.device("cpu")
        obs_seq    = obs_seq.to(cpu)
        action_seq = action_seq.to(cpu)
        concept_ids = concept_ids.to(cpu)
        # Ensure parameters are on CPU too
        if next(self.parameters()).device != cpu:
            self.to(cpu)

        T, B, _ = obs_seq.shape
        device   = cpu

        # Concept embedding: (B, embed_dim) -> broadcast to (T, B, embed_dim)
        c_embed = self.concept_embed(concept_ids)              # (B, embed_dim)
        c_embed = c_embed.unsqueeze(0).expand(T, -1, -1)       # (T, B, embed_dim)

        # Action embedding: (T, B) -> (T, B, embed_dim)
        a_embed = self.action_embed(action_seq)                # (T, B, embed_dim)

        # Concatenate inputs for each timestep
        gru_input = torch.cat([obs_seq, a_embed, c_embed], dim=-1)  # (T, B, *)

        # Encode trajectory
        gru_out, _ = self.gru(gru_input)   # (T, B, gru_hidden)
        final_ctx  = gru_out[-1]           # (B, gru_hidden) — last hidden state

        # Run debate heads
        p_score = self.proponent_head(final_ctx).squeeze(-1)   # (B,)
        o_score = self.opponent_head(final_ctx).squeeze(-1)    # (B,)

        # Determine winner per sample
        winners: List[str] = []
        for p, o in zip(p_score.tolist(), o_score.tolist()):
            if p > o + 0.05:
                winners.append("proponent")
            elif o > p + 0.05:
                winners.append("opponent")
            else:
                winners.append("draw")

        # Shaping reward: +1 when proponent wins, -1 when opponent wins, 0 draw
        win_tensor = torch.tensor(
            [1.0 if w == "proponent" else (-1.0 if w == "opponent" else 0.0)
             for w in winners],
            device=device,
        )
        shaping = (p_score - o_score).clamp(-1.0, 1.0)

        return DebateVerdict(
            proponent_score=p_score,
            opponent_score=o_score,
            winner=winners,
            shaping_reward=shaping,
        )

    # ------------------------------------------------------------------
    def compute_debate_loss(
        self,
        verdict: DebateVerdict,
        game_outcomes: torch.Tensor,  # (B,) float: +1 good, -1 bad (from env reward)
    ) -> torch.Tensor:
        """
        Supervised debate loss:
          - When game outcome is positive (reward > 0), proponent SHOULD win.
          - When game outcome is negative, opponent SHOULD win.

        This uses BCELoss on both heads with soft labels derived from outcome.
        """
        p_target = (game_outcomes > 0).float()   # (B,)
        o_target = (game_outcomes < 0).float()   # (B,)

        p_loss = F.binary_cross_entropy(verdict.proponent_score, p_target)
        o_loss = F.binary_cross_entropy(verdict.opponent_score,  o_target)
        return (p_loss + o_loss) * 0.5
