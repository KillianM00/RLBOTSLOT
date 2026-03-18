"""
Tests for reward components.
All tests use mock game state objects — no rlgym installation needed.
"""
import pytest
import numpy as np
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Mock state builder
# ---------------------------------------------------------------------------

def _vec(x=0, y=0, z=0):
    return SimpleNamespace(x=x, y=y, z=z)


def _make_state(
    car_pos=(0, -1000, 50),
    car_vel=(0, 500, 0),
    ball_pos=(0, 0, 93),
    ball_vel=(0, 1000, 0),
    boost=100.0,
    team=0,
    blue_score=0,
    orange_score=0,
    on_ground=True,
):
    car_physics = SimpleNamespace(
        position=_vec(*car_pos),
        linear_velocity=_vec(*car_vel),
        rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
    )
    car = SimpleNamespace(
        physics=car_physics,
        team_num=team,
        boost_amount=boost,
        match_demolishes=0,
    )
    ball_physics = SimpleNamespace(
        position=_vec(*ball_pos),
        linear_velocity=_vec(*ball_vel),
    )
    last_touch = SimpleNamespace(game_seconds=1.0, player_index=0)
    ball = SimpleNamespace(
        position=_vec(*ball_pos),
        linear_velocity=_vec(*ball_vel),
        physics=ball_physics,
        last_touch=last_touch,
    )
    teams = [
        SimpleNamespace(score=blue_score),
        SimpleNamespace(score=orange_score),
    ]
    state = SimpleNamespace(
        cars={"agent_0": car},
        ball=ball,
        teams=teams,
        game_seconds_elapsed=5.0,
    )
    return state


AGENTS  = ["agent_0"]
SHARED  = {}
TERM    = {"agent_0": False}
TRUNC   = {"agent_0": False}


# ---------------------------------------------------------------------------
# TouchReward
# ---------------------------------------------------------------------------

class TestTouchReward:
    def test_no_touch_returns_zero(self):
        from rewards.touch_reward import TouchReward
        r = TouchReward()
        state = _make_state()
        r.reset(AGENTS, state, SHARED)
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == 0.0

    def test_aerial_multiplier(self):
        from rewards.touch_reward import TouchReward
        r = TouchReward(aerial_multiplier=3.0)
        # Simply check object is initialised correctly
        assert r.aerial_multiplier == 3.0


# ---------------------------------------------------------------------------
# BallControlReward
# ---------------------------------------------------------------------------

class TestBallControlReward:
    def test_ball_far_away_no_reward(self):
        from rewards.ball_control_reward import BallControlReward
        r = BallControlReward()
        state = _make_state(car_pos=(0, 0, 20), ball_pos=(2000, 0, 93))
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == 0.0

    def test_ball_on_roof_gives_reward(self):
        from rewards.ball_control_reward import BallControlReward
        r = BallControlReward()
        # Ball z = 100, car z = 20, horiz_dist = 50
        state = _make_state(car_pos=(0, 0, 20), ball_pos=(30, 0, 140))
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] > 0.0

    def test_reward_in_range(self):
        from rewards.ball_control_reward import BallControlReward
        r = BallControlReward()
        state = _make_state(car_pos=(0, 0, 20), ball_pos=(50, 0, 140))
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert 0.0 <= rewards["agent_0"] <= 1.0


# ---------------------------------------------------------------------------
# GoalScoredReward
# ---------------------------------------------------------------------------

class TestGoalScoredReward:
    def test_no_goal_no_reward(self):
        from rewards.goal_reward import GoalScoredReward
        r = GoalScoredReward()
        state = _make_state(blue_score=0, orange_score=0)
        r.reset(AGENTS, state, SHARED)
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == 0.0

    def test_goal_scored_gives_positive_reward(self):
        from rewards.goal_reward import GoalScoredReward
        r = GoalScoredReward(goal_reward=10.0)
        state_before = _make_state(blue_score=0, orange_score=0)
        r.reset(AGENTS, state_before, SHARED)
        state_after  = _make_state(blue_score=1, orange_score=0)
        rewards = r.get_rewards(AGENTS, state_after, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == pytest.approx(10.0)

    def test_concede_gives_negative_reward(self):
        from rewards.goal_reward import GoalScoredReward
        r = GoalScoredReward(concede_reward=-10.0)
        state_before = _make_state(blue_score=0, orange_score=0, team=0)
        r.reset(AGENTS, state_before, SHARED)
        state_after  = _make_state(blue_score=0, orange_score=1, team=0)
        rewards = r.get_rewards(AGENTS, state_after, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# ShotReward
# ---------------------------------------------------------------------------

class TestShotReward:
    def test_slow_ball_no_reward(self):
        from rewards.shot_reward import ShotReward
        r = ShotReward()
        state = _make_state(ball_vel=(0, 100, 0))  # slow
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == 0.0

    def test_fast_ball_toward_goal_gives_reward(self):
        from rewards.shot_reward import ShotReward
        r = ShotReward()
        # Ball at centre (0,0) moving fast toward orange goal (0, +5120)
        state = _make_state(ball_pos=(0, 0, 93), ball_vel=(0, 2000, 0))
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] > 0.0


# ---------------------------------------------------------------------------
# AntiChaseReward
# ---------------------------------------------------------------------------

class TestAntiChaseReward:
    def test_first_touch_no_penalty(self):
        from rewards.anti_chase_reward import AntiChaseReward
        r = AntiChaseReward()
        state = _make_state()
        r.reset(AGENTS, state, SHARED)
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        # No previous touch recorded, so no penalty
        assert rewards["agent_0"] == 0.0


# ---------------------------------------------------------------------------
# CurriculumProgressReward
# ---------------------------------------------------------------------------

class TestCurriculumProgressReward:
    def test_no_advance_no_reward(self):
        from rewards.curriculum_progress_reward import CurriculumProgressReward
        r = CurriculumProgressReward()
        state = _make_state()
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == 0.0

    def test_advance_gives_bonus(self):
        from rewards.curriculum_progress_reward import CurriculumProgressReward
        r = CurriculumProgressReward()
        r.notify_stage_advance(1)
        state = _make_state()
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards["agent_0"] == pytest.approx(r.STAGE_BONUS)
        # Second call should not give bonus again
        rewards2 = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert rewards2["agent_0"] == 0.0


# ---------------------------------------------------------------------------
# DynamicCombinedReward
# ---------------------------------------------------------------------------

class TestDynamicCombinedReward:
    def test_from_config_stage_0(self):
        from rewards.combined_reward import DynamicCombinedReward
        r = DynamicCombinedReward.from_config(stage=0)
        assert "TouchReward" in r._components
        assert "GoalScoredReward" in r._components

    def test_stage_advance_updates_weights(self):
        from rewards.combined_reward import DynamicCombinedReward
        r = DynamicCombinedReward.from_config(stage=0)
        r.update_weights(3)
        assert r.stage == 3
        assert r._weights.get("AerialReward", 0.0) > 0.0

    def test_get_rewards_returns_dict(self):
        from rewards.combined_reward import DynamicCombinedReward
        r = DynamicCombinedReward.from_config(stage=0)
        state = _make_state()
        r.reset(AGENTS, state, SHARED)
        rewards = r.get_rewards(AGENTS, state, TERM, TRUNC, SHARED)
        assert isinstance(rewards, dict)
        assert "agent_0" in rewards
