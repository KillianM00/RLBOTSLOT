"""
ConceptLibrary: registry for all learned skill concepts.

Each Concept represents a "course" in the professor metaphor:
  - A named skill (e.g. "AerialRecovery_3", "FlickSequence_1")
  - A 128-d embedding that positions it in concept-space
  - A win_rate tracking how often the DebateEngine's Proponent wins
    (i.e. how well the bot executes this concept)
  - Usage counts, source tag, creation timestamp

MAX_N_CONCEPTS slots are pre-allocated in the GatingNetwork so that
new concept registrations never require a network architecture restart.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config as cfg


@dataclass
class Concept:
    concept_id: str
    slot_index: int          # index into GatingNetwork's expert list (0..MAX_N_CONCEPTS-1)
    embedding: np.ndarray    # shape (embed_dim,)
    win_rate: float = 0.5    # proponent win fraction (EMA)
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)
    source: str = "manual"   # "manual" | "replay_discovery" | "curriculum"
    bc_steps_done: int = 0   # imitation steps completed for this concept


class ConceptLibrary:
    """
    Manages the full set of active concepts.

    Key design decisions:
      - Pre-allocates MAX_N_CONCEPTS=64 slots with an active_mask boolean array.
        Slot 0..INITIAL_N_CONCEPTS-1 are active at startup; rest start inactive.
      - New concepts claim the next free slot; the GatingNetwork simply unmasks it.
      - Concepts with persistent win_rate < CONCEPT_MIN_WIN_RATE are pruned
        (slot is freed and masked again).
    """

    def __init__(
        self,
        max_concepts: int = cfg.MAX_N_CONCEPTS,
        embed_dim: int = cfg.EMBED_DIM,
    ) -> None:
        self.max_concepts = max_concepts
        self.embed_dim    = embed_dim
        self._concepts: Dict[str, Concept] = {}       # id -> Concept
        self._slot_to_id: Dict[int, str]   = {}       # slot_index -> concept_id
        self.active_mask: np.ndarray = np.zeros(max_concepts, dtype=bool)

        # Seed initial concepts with random embeddings
        for i in range(cfg.INITIAL_N_CONCEPTS):
            cid = f"InitConcept_{i}"
            embed = np.random.randn(embed_dim).astype(np.float32) * 0.1
            self._register_at_slot(cid, i, embed, source="manual")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, concept_id: str, embedding: np.ndarray, source: str = "replay_discovery") -> Optional[int]:
        """
        Register a new concept.  Returns the slot_index or None if full.
        If a concept with the same id already exists, returns its slot.
        """
        if concept_id in self._concepts:
            return self._concepts[concept_id].slot_index

        free_slot = self._find_free_slot()
        if free_slot is None:
            return None  # all slots occupied; caller should prune first

        self._register_at_slot(concept_id, free_slot, embedding, source)
        return free_slot

    def _register_at_slot(self, concept_id: str, slot: int, embedding: np.ndarray, source: str) -> None:
        c = Concept(
            concept_id=concept_id,
            slot_index=slot,
            embedding=np.asarray(embedding, dtype=np.float32),
            source=source,
        )
        self._concepts[concept_id] = c
        self._slot_to_id[slot] = concept_id
        self.active_mask[slot] = True

    def _find_free_slot(self) -> Optional[int]:
        for i in range(self.max_concepts):
            if not self.active_mask[i]:
                return i
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, concept_id: str) -> Optional[Concept]:
        return self._concepts.get(concept_id)

    def get_by_slot(self, slot: int) -> Optional[Concept]:
        cid = self._slot_to_id.get(slot)
        return self._concepts.get(cid) if cid else None

    def all_active(self) -> List[Concept]:
        return [c for c in self._concepts.values() if self.active_mask[c.slot_index]]

    def get_active_slots(self) -> List[int]:
        return [i for i in range(self.max_concepts) if self.active_mask[i]]

    def n_active(self) -> int:
        return int(self.active_mask.sum())

    def find_nearest(self, embedding: np.ndarray, k: int = 5) -> List[Concept]:
        """Return the k concepts with embeddings most similar (cosine) to `embedding`."""
        active = self.all_active()
        if not active:
            return []
        embeds = np.stack([c.embedding for c in active])  # (n, embed_dim)
        query  = embedding / (np.linalg.norm(embedding) + 1e-8)
        norms  = np.linalg.norm(embeds, axis=1, keepdims=True)
        normed = embeds / (norms + 1e-8)
        sims   = normed @ query
        top_k  = np.argsort(sims)[::-1][:k]
        return [active[i] for i in top_k]

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_win_rate(self, concept_id: str, proponent_won: bool, alpha: float = 0.05) -> None:
        """EMA update of win_rate for one debate outcome."""
        c = self._concepts.get(concept_id)
        if c is None:
            return
        outcome = 1.0 if proponent_won else 0.0
        c.win_rate = (1 - alpha) * c.win_rate + alpha * outcome
        c.usage_count += 1

    def record_bc_steps(self, concept_id: str, steps: int) -> None:
        c = self._concepts.get(concept_id)
        if c:
            c.bc_steps_done += steps

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, min_win_rate: float = cfg.CONCEPT_MIN_WIN_RATE, min_usage: int = 50) -> List[str]:
        """
        Remove concepts with low win_rate after sufficient usage.
        Returns list of pruned concept_ids.
        """
        pruned = []
        for cid, c in list(self._concepts.items()):
            if c.usage_count >= min_usage and c.win_rate < min_win_rate:
                self.active_mask[c.slot_index] = False
                del self._concepts[cid]
                del self._slot_to_id[c.slot_index]
                pruned.append(cid)
        return pruned

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ConceptLibrary":
        with open(path, "rb") as f:
            return pickle.load(f)
