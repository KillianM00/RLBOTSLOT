"""
WorkerPool: thin wrapper that configures rlgym-ppo's Learner with
the correct number of worker processes and our custom env factory.
"""
from __future__ import annotations

import functools
from typing import Optional

import config as cfg


def make_env_fn(stage: int = 0):
    """Return a zero-argument callable that creates an env at the given stage."""
    from engine.env_factory import build_env
    return functools.partial(build_env, curriculum_stage=stage)


def build_learner(
    stage: int = 0,
    n_proc: Optional[int] = None,
    logger=None,
    checkpoint_folder: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    """
    Instantiate an rlgym-ppo Learner with our custom env factory.

    Returns rlgym_ppo.Learner or None if rlgym-ppo is not installed.
    """
    try:
        from rlgym_ppo import Learner  # type: ignore
    except ImportError:
        if logger:
            logger.warning("rlgym-ppo not installed — Learner unavailable.")
        return None

    import os
    n_proc    = n_proc or cfg.N_WORKERS
    env_fn    = make_env_fn(stage)
    # Separate folder for rlgym-ppo's own checkpoints (saves as numeric subfolders)
    ckpt_dir  = checkpoint_folder or os.path.join(str(cfg.CHECKPOINT_DIR), "rlgym_ppo")
    os.makedirs(ckpt_dir, exist_ok=True)

    return Learner(
        env_create_function    = env_fn,
        n_proc                 = n_proc,
        timestep_limit         = cfg.TS_PER_ITERATION,   # learn() stops after one iteration
        ts_per_iteration       = cfg.TS_PER_ITERATION,
        exp_buffer_size        = cfg.TS_PER_ITERATION * 3,
        ppo_epochs             = cfg.PPO_EPOCHS,
        ppo_batch_size         = cfg.PPO_BATCH_SIZE,
        ppo_minibatch_size     = cfg.PPO_MINIBATCH_SIZE,
        policy_lr              = cfg.POLICY_LR,
        critic_lr              = cfg.CRITIC_LR,
        ppo_clip_range         = cfg.CLIP_RANGE,
        ppo_ent_coef           = cfg.PPO_ENT_COEF,
        gae_lambda             = cfg.GAE_LAMBDA,
        gae_gamma              = cfg.GAMMA,
        standardize_returns    = cfg.STANDARDIZE_RETURNS,
        policy_layer_sizes     = cfg.EXPERT_HIDDEN,       # (512, 256)
        critic_layer_sizes     = (512, 256, 128),
        checkpoints_save_folder= ckpt_dir,
        add_unix_timestamp     = False,   # fixed folder so next Learner loads previous weights
        log_to_wandb           = wandb_project is not None,
        wandb_project_name     = wandb_project,
        wandb_run_name         = wandb_run_name,
        checkpoint_load_folder = "latest",
        device                 = "cpu",     # DirectML not compatible with rlgym-ppo
    )
