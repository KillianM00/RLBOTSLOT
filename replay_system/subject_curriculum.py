"""
SubjectCurriculum: bridges replay discovery with the professor system.

After BehaviorClusterer identifies new subjects:
  1. Register each subject as a Concept in ConceptLibrary
  2. Trigger ImitationLearner to pre-train the corresponding ExpertNetwork
  3. Notify ProfessorPolicy to sync its active_mask
  4. Log the discovery

This creates the feedback loop:
  replays → subjects → concepts → experts → training
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg
from professor.concept_library import ConceptLibrary
from replay_system.parser import Trajectory

logger = logging.getLogger(__name__)


class SubjectCurriculum:
    """
    Manages the injection of replay-discovered subjects into the professor system.
    """

    def __init__(
        self,
        concept_library: ConceptLibrary,
        professor_policy=None,   # ProfessorPolicy; injected after construction
    ) -> None:
        self.concept_library  = concept_library
        self.professor_policy = professor_policy

    def set_professor_policy(self, policy) -> None:
        self.professor_policy = policy

    # ------------------------------------------------------------------
    def register_subjects(
        self,
        subject_names: Dict[int, str],                    # cluster_id -> name
        exemplars: Dict[int, Trajectory],                 # cluster_id -> exemplar
        cluster_trajectories: Dict[int, List[Trajectory]], # cluster_id -> all trajs
    ) -> List[Tuple[str, int]]:
        """
        Register newly discovered subjects.
        Returns list of (subject_name, slot_index) for newly registered ones.
        """
        registered: List[Tuple[str, int]] = []

        for cid, name in subject_names.items():
            if name in [c.concept_id for c in self.concept_library.all_active()]:
                logger.debug("Subject %s already registered — skipping.", name)
                continue

            # Build a concept embedding from the exemplar's mean feature vector
            exemplar = exemplars.get(cid)
            if exemplar is None:
                continue

            embedding = self._embed_trajectory(exemplar)
            slot = self.concept_library.register(
                concept_id=name,
                embedding=embedding,
                source="replay_discovery",
            )
            if slot is None:
                logger.warning("ConceptLibrary full — could not register %s.", name)
                continue

            registered.append((name, slot))
            logger.info("Registered new subject '%s' at slot %d", name, slot)

        # Notify ProfessorPolicy to update its active_mask
        if registered and self.professor_policy is not None:
            self.professor_policy.notify_concept_update()

        return registered

    # ------------------------------------------------------------------
    def _embed_trajectory(self, traj: Trajectory) -> np.ndarray:
        """
        Create a concept embedding from a trajectory's feature vector.
        Projects FEATURE_DIM (12) → EMBED_DIM (128) via random projection.
        (A learned encoder can replace this in future.)
        """
        from replay_system.clusterer import BehaviorClusterer
        feat    = BehaviorClusterer().extract_features(traj)   # (12,)
        rng     = np.random.default_rng(abs(hash(traj.replay_id)) % (2**32))
        proj    = rng.standard_normal((len(feat), cfg.EMBED_DIM)).astype(np.float32)
        proj   /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
        embed   = feat @ proj                                   # (embed_dim,)
        embed  /= np.linalg.norm(embed) + 1e-8
        return embed
