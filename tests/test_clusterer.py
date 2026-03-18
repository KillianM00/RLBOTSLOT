"""
Tests for BehaviorClusterer, SubjectNamer, and ConceptLibrary.
No rlgym or network required.
"""
import numpy as np
import pytest

from replay_system.parser import Trajectory
from replay_system.clusterer import BehaviorClusterer
from replay_system.subject_namer import SubjectNamer
from professor.concept_library import ConceptLibrary, Concept


def _make_trajectory(aerial=False, n_frames=30):
    frames = np.zeros((n_frames, 16), dtype=np.float32)
    if aerial:
        frames[:, 2] = 0.5   # car_z (normalised, ~1022 uu)
        frames[:, 8] = 0.6   # ball_z (normalised, ~1227 uu)
    else:
        frames[:, 2] = 0.01  # near ground
        frames[:, 8] = 0.05
    frames[:, 12] = 0.5   # boost
    frames[:, 13] = 1.0 if not aerial else 0.0   # on_ground
    actions = np.zeros(n_frames, dtype=np.int32)
    return Trajectory("test_replay", "player", frames, actions)


class TestBehaviorClusterer:
    def test_extract_features_shape(self):
        c    = BehaviorClusterer()
        traj = _make_trajectory(aerial=False)
        feat = c.extract_features(traj)
        assert feat.shape == (c.FEATURE_DIM,)
        assert not np.isnan(feat).any()

    def test_aerial_vs_ground_features_differ(self):
        c      = BehaviorClusterer()
        aerial = c.extract_features(_make_trajectory(aerial=True))
        ground = c.extract_features(_make_trajectory(aerial=False))
        # Aerial fraction (index 3) should be higher for aerial trajectories
        assert aerial[3] > ground[3]

    def test_fit_returns_correct_length(self):
        c     = BehaviorClusterer(min_cluster_size=2, min_samples=1)
        trajs = [_make_trajectory(aerial=(i % 2 == 0)) for i in range(20)]
        labels = c.fit(trajs)
        assert len(labels) == 20

    def test_exemplar_returned_for_each_cluster(self):
        c     = BehaviorClusterer(min_cluster_size=2, min_samples=1)
        trajs = [_make_trajectory(aerial=(i % 2 == 0)) for i in range(20)]
        labels = c.fit(trajs)
        exemplars = c.get_cluster_exemplars(labels, trajs)
        for cid, ex in exemplars.items():
            assert isinstance(ex, Trajectory)


class TestSubjectNamer:
    def test_name_is_string_with_cluster_id(self):
        namer = SubjectNamer()
        feat  = np.zeros(12, dtype=np.float32)
        feat[3] = 0.9  # high aerial fraction -> AerialRecovery or CeilingShot
        name  = namer.name_cluster(7, _make_trajectory(), feat)
        assert isinstance(name, str)
        assert "7" in name

    def test_names_are_unique(self):
        namer = SubjectNamer()
        feat  = np.zeros(12, dtype=np.float32)
        names = {namer.name_cluster(i, _make_trajectory(), feat) for i in range(10)}
        assert len(names) == 10

    def test_aerial_feature_picks_aerial_prefix(self):
        namer = SubjectNamer()
        feat  = np.zeros(12, dtype=np.float32)
        feat[3] = 0.85   # aerial_frac
        feat[1] = 0.55   # max_ball_z
        name  = namer.name_cluster(0, _make_trajectory(), feat)
        assert "Ceiling" in name or "Aerial" in name


class TestConceptLibrary:
    def test_initial_concepts_active(self):
        import config as cfg
        lib = ConceptLibrary()
        assert lib.n_active() == cfg.INITIAL_N_CONCEPTS

    def test_register_new_concept(self):
        lib   = ConceptLibrary()
        embed = np.random.randn(128).astype(np.float32)
        slot  = lib.register("TestConcept", embed, source="manual")
        assert slot is not None
        assert lib.get("TestConcept") is not None

    def test_prune_low_win_rate(self):
        lib   = ConceptLibrary()
        embed = np.random.randn(128).astype(np.float32)
        lib.register("WeakConcept", embed)
        concept = lib.get("WeakConcept")
        # Set very low win rate with sufficient usage
        concept.win_rate   = 0.02
        concept.usage_count = 200
        pruned = lib.prune(min_win_rate=0.05, min_usage=50)
        assert "WeakConcept" in pruned
        assert lib.get("WeakConcept") is None

    def test_find_nearest_returns_similar(self):
        lib = ConceptLibrary()
        concepts = lib.all_active()
        if not concepts:
            return
        target = concepts[0].embedding.copy()
        results = lib.find_nearest(target, k=3)
        assert len(results) <= 3
        # The most similar should be the target itself (or very close)
        assert results[0].concept_id == concepts[0].concept_id

    def test_active_mask_sync(self):
        lib = ConceptLibrary()
        n = lib.n_active()
        embed = np.random.randn(128).astype(np.float32)
        lib.register("NewOne", embed)
        assert lib.n_active() == n + 1
