"""
ConceptEmbeddingNetwork: maps concept_id integers to dense embeddings.
Embeddings are used by:
  - DebateEngine (conditions debate on concept)
  - GatingNetwork (computes similarity between obs context and concept embeddings)
  - ConceptLibrary (stored as numpy arrays for similarity search)
"""
from __future__ import annotations

import torch
import torch.nn as nn

import config as cfg


class ConceptEmbeddingNetwork(nn.Module):
    """
    Learnable embedding table: concept_id -> embed_dim vector.
    Pre-allocated for MAX_N_CONCEPTS slots; unused slots are masked.
    """

    def __init__(
        self,
        max_concepts: int = cfg.MAX_N_CONCEPTS,
        embed_dim: int = cfg.EMBED_DIM,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_concepts, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, concept_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        concept_ids : (batch,) long tensor

        Returns
        -------
        embeddings : (batch, embed_dim)
        """
        return self.embed(concept_ids)

    def get_all_active(self, active_mask: torch.Tensor) -> torch.Tensor:
        """
        Return embeddings for all active concept slots.

        Parameters
        ----------
        active_mask : (max_concepts,) bool tensor

        Returns
        -------
        active_embeds : (n_active, embed_dim)
        """
        indices = active_mask.nonzero(as_tuple=True)[0]
        return self.embed(indices)
