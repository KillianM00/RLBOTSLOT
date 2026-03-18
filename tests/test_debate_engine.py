"""
Tests for the DebateEngine and GatingNetwork.
Requires PyTorch but NOT rlgym.
"""
import pytest
import torch

import config as cfg


class TestDebateEngine:
    def test_forward_shape(self):
        from professor.debate_engine import DebateEngine
        engine = DebateEngine(
            obs_dim=cfg.OBS_DIM,
            n_actions=cfg.N_ACTIONS,
            gru_hidden=64,
            embed_dim=32,
            max_concepts=8,
        )
        T, B = cfg.DEBATE_SEQ_LEN, 4
        obs_seq     = torch.randn(T, B, cfg.OBS_DIM)
        action_seq  = torch.randint(0, cfg.N_ACTIONS, (T, B))
        concept_ids = torch.zeros(B, dtype=torch.long)

        verdict = engine(obs_seq, action_seq, concept_ids)
        assert verdict.proponent_score.shape == (B,)
        assert verdict.opponent_score.shape  == (B,)
        assert len(verdict.winner) == B
        assert verdict.shaping_reward.shape  == (B,)

    def test_scores_in_zero_one(self):
        from professor.debate_engine import DebateEngine
        engine = DebateEngine(obs_dim=cfg.OBS_DIM, gru_hidden=64, embed_dim=32, max_concepts=8)
        T, B = 10, 2
        obs_seq     = torch.randn(T, B, cfg.OBS_DIM)
        action_seq  = torch.randint(0, cfg.N_ACTIONS, (T, B))
        concept_ids = torch.zeros(B, dtype=torch.long)
        verdict = engine(obs_seq, action_seq, concept_ids)
        assert (verdict.proponent_score >= 0).all() and (verdict.proponent_score <= 1).all()
        assert (verdict.opponent_score  >= 0).all() and (verdict.opponent_score  <= 1).all()

    def test_compute_debate_loss_is_scalar(self):
        from professor.debate_engine import DebateEngine
        engine = DebateEngine(obs_dim=cfg.OBS_DIM, gru_hidden=64, embed_dim=32, max_concepts=8)
        T, B = 10, 4
        obs_seq     = torch.randn(T, B, cfg.OBS_DIM)
        action_seq  = torch.randint(0, cfg.N_ACTIONS, (T, B))
        concept_ids = torch.zeros(B, dtype=torch.long)
        outcomes    = torch.randn(B)
        verdict = engine(obs_seq, action_seq, concept_ids)
        loss    = engine.compute_debate_loss(verdict, outcomes)
        assert loss.ndim == 0   # scalar
        assert not torch.isnan(loss)

    def test_winner_string_valid(self):
        from professor.debate_engine import DebateEngine
        engine = DebateEngine(obs_dim=cfg.OBS_DIM, gru_hidden=64, embed_dim=32, max_concepts=8)
        T, B = 10, 6
        obs_seq     = torch.randn(T, B, cfg.OBS_DIM)
        action_seq  = torch.randint(0, cfg.N_ACTIONS, (T, B))
        concept_ids = torch.zeros(B, dtype=torch.long)
        verdict = engine(obs_seq, action_seq, concept_ids)
        for w in verdict.winner:
            assert w in ("proponent", "opponent", "draw")


class TestGatingNetwork:
    def test_output_shapes(self):
        from professor.gating_network import GatingNetwork
        gating = GatingNetwork(obs_dim=cfg.OBS_DIM, max_concepts=16, k=2, hidden_sizes=(64,))
        gating.active_mask[:8] = True
        obs = torch.randn(8, cfg.OBS_DIM)
        gate_weights, top_k_indices, lb_loss = gating(obs)
        assert gate_weights.shape  == (8, 16)
        assert top_k_indices.shape == (8, 2)
        assert lb_loss.ndim == 0

    def test_inactive_slots_zeroed(self):
        from professor.gating_network import GatingNetwork
        gating = GatingNetwork(obs_dim=cfg.OBS_DIM, max_concepts=16, k=2, hidden_sizes=(64,))
        gating.active_mask[:4] = True   # only first 4 active
        obs = torch.randn(4, cfg.OBS_DIM)
        gate_weights, _, _ = gating(obs)
        # Inactive slots (4-15) should have zero weight
        assert (gate_weights[:, 4:] == 0.0).all()

    def test_gate_weights_sum_to_one_approx(self):
        from professor.gating_network import GatingNetwork
        gating = GatingNetwork(obs_dim=cfg.OBS_DIM, max_concepts=16, k=2, hidden_sizes=(64,))
        gating.active_mask[:8] = True
        obs = torch.randn(8, cfg.OBS_DIM)
        gate_weights, _, _ = gating(obs)
        sums = gate_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)


class TestProfessorPolicy:
    def test_get_action_and_value_shape(self):
        from professor.concept_library import ConceptLibrary
        from professor.professor_policy import ProfessorPolicy
        lib    = ConceptLibrary(max_concepts=16)
        policy = ProfessorPolicy(lib, obs_dim=cfg.OBS_DIM, max_concepts=16, k=2)
        obs    = torch.randn(4, cfg.OBS_DIM)
        action, log_prob, entropy, value = policy.get_action_and_value(obs)
        assert action.shape   == (4,)
        assert log_prob.shape == (4,)
        assert entropy.ndim   == 0
        assert value.shape    == (4,)

    def test_actions_in_valid_range(self):
        from professor.concept_library import ConceptLibrary
        from professor.professor_policy import ProfessorPolicy
        lib    = ConceptLibrary(max_concepts=16)
        policy = ProfessorPolicy(lib, obs_dim=cfg.OBS_DIM, max_concepts=16, k=2)
        obs    = torch.randn(32, cfg.OBS_DIM)
        action, _, _, _ = policy.get_action_and_value(obs)
        assert (action >= 0).all() and (action < cfg.N_ACTIONS).all()
