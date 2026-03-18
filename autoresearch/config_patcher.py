"""
Safe config mutation for autoresearch experiments.

Snapshots, mutates, and restores the `config` module's attributes.
Since build_learner() reads cfg.* at call time, patching before the call
is sufficient for the new Learner to pick up the mutation.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Set, Tuple

import config as cfg

logger = logging.getLogger("autoresearch.config_patcher")

# Attributes that are NEVER mutated (structural / fixed-shape)
FROZEN_ATTRS: Set[str] = {
    "BASE_DIR", "CHECKPOINT_DIR", "REPLAY_CACHE_DIR", "REPLAY_DB_PATH",
    "MEMMAP_PATH", "LOG_DIR", "CONCEPT_SAVE_PATH",
    "N_ACTIONS", "OBS_DIM", "MAX_N_CONCEPTS",
    "DEVICE", "WANDB_PROJECT", "WANDB_ENTITY",
    "SUBJECT_PREFIXES",
}

# All scalar/dict attrs in config.py that are safe to mutate
MUTABLE_ATTRS: Set[str] = {
    # PPO
    "POLICY_LR", "CRITIC_LR", "PPO_ENT_COEF", "CLIP_RANGE", "GAE_LAMBDA",
    "GAMMA", "MAX_GRAD_NORM", "PPO_EPOCHS", "PPO_BATCH_SIZE", "PPO_MINIBATCH_SIZE",
    "STANDARDIZE_RETURNS", "TS_PER_ITERATION",
    # MoE architecture
    "EXPERT_HIDDEN", "GATING_HIDDEN", "EMBED_DIM", "K_EXPERTS",
    "INITIAL_N_CONCEPTS", "GATING_NOISE_STD",
    # Aux loss
    "LOAD_BALANCE_COEF", "DEBATE_LOSS_COEF", "BC_REGULARIZE_COEF",
    # Debate
    "DEBATE_GRU_HIDDEN", "DEBATE_SEQ_LEN", "MAX_DEBATES_PER_ITER", "MIN_CONCEPT_USAGE",
    # Rewards
    "REWARD_WEIGHTS",
    "GOAL_REWARD", "CONCEDE_REWARD", "TOUCH_AERIAL_MULTIPLIER",
    "FLICK_VELOCITY_THRESHOLD", "ANTI_CHASE_SAME_TOUCH_PENALTY",
    "ANTI_CHASE_OVERCOMMIT_PENALTY", "BOOST_LOW_THRESHOLD", "DEMO_MIN_DISTANCE",
    # Curriculum
    "CURRICULUM_THRESHOLDS",
    # Episode
    "MAX_EPISODE_SECONDS", "NO_TOUCH_TIMEOUT",
    # Replay
    "REPLAY_FETCH_INTERVAL", "REPLAY_BATCH_SIZE", "REPLAY_RATE_LIMIT_RPS",
    "SEGMENT_LENGTH", "HDBSCAN_MIN_CLUSTER", "HDBSCAN_MIN_SAMPLES",
    "BC_EPOCHS", "BC_BATCH_SIZE", "BC_REGULARIZE_STEPS", "CONCEPT_MIN_WIN_RATE",
    # Optimization
    "COMPILE_MODE", "USE_AMP", "REPLAY_BUFFER_CAPACITY", "MIN_INFERENCE_BATCH",
    # Logging
    "LOG_INTERVAL", "CHECKPOINT_INTERVAL",
}

# Params whose mutation changes network shapes (requires fresh model)
SHAPE_CHANGING: Set[str] = {
    "EXPERT_HIDDEN", "GATING_HIDDEN", "EMBED_DIM", "K_EXPERTS",
    "DEBATE_GRU_HIDDEN",
}


def snapshot() -> Dict[str, Any]:
    """Deep-copy all mutable cfg attributes into a dict."""
    snap = {}
    for attr in MUTABLE_ATTRS:
        if hasattr(cfg, attr):
            snap[attr] = copy.deepcopy(getattr(cfg, attr))
    return snap


def restore(snap: Dict[str, Any]) -> None:
    """Restore cfg from a snapshot dict."""
    for attr, value in snap.items():
        setattr(cfg, attr, copy.deepcopy(value))
    logger.debug("Config restored from snapshot (%d attrs)", len(snap))


def apply_mutations(mutations: Dict[str, Any]) -> None:
    """
    Apply mutations to cfg. Supports dot-notation for nested dicts:
      "POLICY_LR"                  -> cfg.POLICY_LR = value
      "REWARD_WEIGHTS.0.TouchReward" -> cfg.REWARD_WEIGHTS[0]["TouchReward"] = value
      "CURRICULUM_THRESHOLDS.1.goal_rate" -> cfg.CURRICULUM_THRESHOLDS[1]["goal_rate"] = value
    """
    for key, value in mutations.items():
        parts = key.split(".")
        if len(parts) == 1:
            # Simple attribute
            if parts[0] in FROZEN_ATTRS:
                logger.warning("Refusing to mutate frozen attr: %s", parts[0])
                continue
            setattr(cfg, parts[0], value)
            logger.info("Mutated cfg.%s = %s", parts[0], value)
        else:
            # Nested: e.g. "REWARD_WEIGHTS.0.TouchReward"
            root_attr = parts[0]
            if root_attr in FROZEN_ATTRS:
                logger.warning("Refusing to mutate frozen attr: %s", root_attr)
                continue
            obj = getattr(cfg, root_attr)
            for part in parts[1:-1]:
                # Try int key first (for dict with int keys), then string
                try:
                    obj = obj[int(part)]
                except (ValueError, KeyError, TypeError):
                    obj = obj[part]
            # Set the final value
            final_key = parts[-1]
            try:
                obj[int(final_key)] = value
            except (ValueError, TypeError):
                obj[final_key] = value
            logger.info("Mutated cfg.%s = %s", key, value)


def mutations_require_fresh_model(mutations: Dict[str, Any]) -> bool:
    """Check if any mutation changes a shape-defining parameter."""
    for key in mutations:
        root = key.split(".")[0]
        if root in SHAPE_CHANGING:
            return True
    return False


def get_current_value(key: str) -> Any:
    """Read a potentially nested config value using dot-notation."""
    parts = key.split(".")
    obj = getattr(cfg, parts[0])
    for part in parts[1:]:
        try:
            obj = obj[int(part)]
        except (ValueError, KeyError, TypeError):
            obj = obj[part]
    return obj
