"""
Tests for curriculum scheduler and action space (no rlgym required).
"""
import numpy as np
import pytest

from engine.curriculum import CurriculumScheduler, TrainingMetrics, STAGES
from engine.action_space import ActionTable, LOOKUP_TABLE, N_ACTIONS


class TestCurriculumScheduler:
    def test_starts_at_stage_zero(self):
        sched = CurriculumScheduler()
        assert sched.current_stage == 0

    def test_should_not_advance_without_metrics(self):
        sched   = CurriculumScheduler()
        metrics = TrainingMetrics()
        assert not sched.should_advance(metrics)

    def test_advances_when_thresholds_met(self):
        sched   = CurriculumScheduler()
        metrics = TrainingMetrics(
            total_timesteps=10_000_000,  # > 5M threshold for stage 1
            goal_rate=0.5,               # > 0.3 threshold
        )
        assert sched.should_advance(metrics)
        sched.advance()
        assert sched.current_stage == 1

    def test_does_not_exceed_max_stage(self):
        sched = CurriculumScheduler()
        for _ in range(20):
            sched.advance()
        assert sched.current_stage == len(STAGES) - 1

    def test_advance_callback_called(self):
        sched = CurriculumScheduler()
        received = []
        sched.register_advance_callback(lambda s: received.append(s))
        sched.advance()
        assert received == [1]

    def test_reward_weights_returned(self):
        sched   = CurriculumScheduler()
        weights = sched.get_reward_weights()
        assert isinstance(weights, dict)
        assert "GoalScoredReward" in weights

    def test_state_dict_roundtrip(self):
        sched = CurriculumScheduler()
        sched.advance()
        sched.advance()
        d = sched.state_dict()
        sched2 = CurriculumScheduler()
        sched2.load_state_dict(d)
        assert sched2.current_stage == sched.current_stage


class TestActionSpace:
    def test_table_shape(self):
        assert LOOKUP_TABLE.shape == (N_ACTIONS, 8)
        assert N_ACTIONS == 90

    def test_get_action_returns_8d_vector(self):
        a = ActionTable.get_action(0)
        assert a.shape == (8,)

    def test_nearest_index_returns_valid_index(self):
        raw = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        idx = ActionTable.nearest_index(raw)
        assert 0 <= idx < N_ACTIONS

    def test_identity_mapping(self):
        """The nearest index of a lookup table action should be itself (or equal entry)."""
        for i in range(N_ACTIONS):
            action = LOOKUP_TABLE[i]
            idx    = ActionTable.nearest_index(action)
            # Result may differ if table has duplicate rows, but should be valid
            assert 0 <= idx < N_ACTIONS
            np.testing.assert_allclose(
                LOOKUP_TABLE[idx], action, atol=0.01,
                err_msg=f"Action {i} does not map to itself"
            )

    def test_zero_action_handled(self):
        raw = np.zeros(8, dtype=np.float32)
        idx = ActionTable.nearest_index(raw)
        assert 0 <= idx < N_ACTIONS


class TestTrainingMetrics:
    def test_update_increments_timesteps(self):
        m = TrainingMetrics()
        m.update(1000, 0.5, 1.0)
        assert m.total_timesteps == 1000

    def test_goal_rate_ema_converges(self):
        m = TrainingMetrics()
        for _ in range(200):
            m.update(1, 1.0, 1.0)
        # After many updates with goal_rate=1.0, should be close to 1.0
        assert m.goal_rate > 0.9
