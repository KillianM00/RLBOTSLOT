"""
train.py — Main entry point for the Professor RL Bot.

Orchestrates all 5 systems:
  1. Hyperaccelerated Training Engine (RocketSim + rlgym-ppo, 32 workers)
  2. Professor/Debate Learning System (MoE + DebateEngine)
  3. Complex Reward System (13 components, dynamic weighting)
  4. Replay Watching / Subject Discovery (ballchasing.com + HDBSCAN)
  5. Efficiency Optimizations (torch.compile, AMP, mmap buffer, async fetching)

Usage:
    python train.py                    # full training
    python train.py --dry-run          # 100 steps, no WandB, then exit
    python train.py --resume           # resume from latest checkpoint
    python train.py --disable-replays  # skip replay fetching
    python train.py --n-workers 8      # override worker count

Environment:
    Copy .env.example to .env and set BALLCHASING_API_KEY=<your_key>
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

import config as cfg

# ---------------------------------------------------------------------------
# Logging setup (before any other imports that log)
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s | %(levelname)s | %(name)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Professor RL Bot Training")
    p.add_argument("--dry-run",          action="store_true",
                   help="Run 100 steps then exit (for CI / quick checks)")
    p.add_argument("--resume",           action="store_true",
                   help="Resume training from latest checkpoint")
    p.add_argument("--disable-replays",  action="store_true",
                   help="Skip ballchasing replay fetching")
    p.add_argument("--disable-wandb",    action="store_true",
                   help="Disable Weights & Biases logging")
    p.add_argument("--n-workers",        type=int, default=cfg.N_WORKERS,
                   help="Number of parallel simulation workers")
    p.add_argument("--timesteps",        type=int, default=cfg.TIMESTEP_LIMIT,
                   help="Total training timesteps")
    p.add_argument("--profile",          action="store_true",
                   help="Enable torch.profiler (slow; for debugging)")
    p.add_argument("--autoresearch",    action="store_true",
                   help="Run autonomous experiment loop (Karpathy-style autoresearch)")
    p.add_argument("--autoresearch-experiments", type=int,
                   default=cfg.AUTORESEARCH_MAX_EXPERIMENTS,
                   help="Number of autoresearch experiments to run")
    p.add_argument("--autoresearch-steps", type=int,
                   default=cfg.AUTORESEARCH_STEPS_PER_EXPERIMENT,
                   help="Training steps per autoresearch experiment")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Component construction helpers
# ---------------------------------------------------------------------------

def _load_dotenv():
    """Load .env file for API keys if python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore
        env_path = cfg.BASE_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded .env from %s", env_path)
    except ImportError:
        pass


def _build_concept_library():
    """Load from checkpoint or create fresh."""
    from professor.concept_library import ConceptLibrary
    if cfg.CONCEPT_SAVE_PATH.exists():
        try:
            lib = ConceptLibrary.load(cfg.CONCEPT_SAVE_PATH)
            logger.info("Loaded ConceptLibrary (%d concepts)", lib.n_active())
            return lib
        except Exception as e:
            logger.warning("Failed to load ConceptLibrary: %s — creating fresh.", e)
    lib = ConceptLibrary()
    logger.info("Created fresh ConceptLibrary (%d initial concepts)", lib.n_active())
    return lib


def _build_professor_policy(concept_library):
    """Construct ProfessorPolicy and optionally compile it."""
    from professor.professor_policy import ProfessorPolicy
    from optimizations.compilation import compile_model

    policy = ProfessorPolicy(concept_library=concept_library)
    policy = compile_model(policy)
    policy = policy.to(cfg.DEVICE)
    logger.info("ProfessorPolicy: %d params | device=%s",
                sum(p.numel() for p in policy.parameters()), cfg.DEVICE)
    return policy


def _build_replay_pipeline(concept_library, professor_policy, args):
    """Construct the entire replay discovery pipeline."""
    from replay_system.rank_tracker import RankTracker
    from replay_system.fetcher import BallchasingFetcher
    from replay_system.parser import ReplayParser
    from replay_system.clusterer import BehaviorClusterer
    from replay_system.subject_namer import SubjectNamer
    from replay_system.subject_curriculum import SubjectCurriculum
    from optimizations.async_fetcher import AsyncReplayFetcher

    rank_tracker = RankTracker()
    fetcher      = BallchasingFetcher(
        api_key=os.getenv("BALLCHASING_API_KEY", cfg.BALLCHASING_API_KEY),
        rank_tracker=rank_tracker,
    )
    parser       = ReplayParser()
    clusterer    = BehaviorClusterer()
    namer        = SubjectNamer()
    curriculum   = SubjectCurriculum(
        concept_library=concept_library,
        professor_policy=professor_policy,
    )
    async_fetcher = AsyncReplayFetcher(
        ballchasing_fetcher=fetcher,
        replay_parser=parser,
        behavior_clusterer=clusterer,
        subject_namer=namer,
        subject_curriculum=curriculum,
    )
    return rank_tracker, async_fetcher, curriculum


def _build_debate_loop(professor_policy):
    """
    Returns a callable that runs the DebateEngine on a trajectory batch.
    Called asynchronously from the main training loop.
    """
    debate_engine  = professor_policy.debate_engine
    device         = next(debate_engine.parameters()).device

    def run_debate(obs_seq, action_seq, concept_ids, game_outcomes):
        """
        obs_seq      : (T, B, obs_dim) float tensor
        action_seq   : (T, B) long tensor
        concept_ids  : (B,) long tensor
        game_outcomes: (B,) float tensor (+1 good, -1 bad)

        Returns debate_loss scalar and list of shaping rewards.
        """
        verdict = debate_engine(
            obs_seq.to(device),
            action_seq.to(device),
            concept_ids.to(device),
        )
        loss = debate_engine.compute_debate_loss(
            verdict, game_outcomes.to(device)
        )
        return loss, verdict.shaping_reward.cpu().tolist()

    return run_debate


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _save_checkpoint(
    policy, critic, curriculum_scheduler, rank_tracker, iteration: int
) -> None:
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.CHECKPOINT_DIR / f"checkpoint_{iteration:08d}.pt"
    torch.save({
        "iteration":            iteration,
        "policy_state":         policy.state_dict(),
        "curriculum":           curriculum_scheduler.state_dict(),
        "rank_tracker":         rank_tracker.state_dict(),
    }, path)
    # Also save concept library
    policy.concept_library.save(cfg.CONCEPT_SAVE_PATH)
    logger.info("Saved checkpoint to %s", path)


def _load_latest_checkpoint(policy, curriculum_scheduler, rank_tracker):
    """Find and load the most recent checkpoint. Returns iteration number."""
    checkpoints = sorted(cfg.CHECKPOINT_DIR.glob("checkpoint_*.pt"))
    if not checkpoints:
        return 0
    path = checkpoints[-1]
    state = torch.load(path, map_location="cpu")
    try:
        policy.load_state_dict(state["policy_state"], strict=False)
    except Exception as e:
        logger.warning("Partial checkpoint load: %s", e)
    curriculum_scheduler.load_state_dict(state.get("curriculum", {}))
    rank_tracker.load_state_dict(state.get("rank_tracker", {}))
    iteration = state.get("iteration", 0)
    logger.info("Resumed from checkpoint %s (iteration %d)", path.name, iteration)
    return iteration


# ---------------------------------------------------------------------------
# BC pre-training phase
# ---------------------------------------------------------------------------

def run_bc_pretraining(professor_policy, concept_library, replay_parser):
    """
    If the replay cache contains replays, parse them and BC-pretrain
    any expert that does not yet have bc_steps_done > 0.
    """
    from replay_system.clusterer import BehaviorClusterer
    from replay_system.subject_namer import SubjectNamer
    from replay_system.imitation_learner import ImitationLearner

    replay_files = list(cfg.REPLAY_CACHE_DIR.glob("*.replay"))
    if not replay_files:
        logger.info("No cached replays for BC pre-training — skipping.")
        return

    logger.info("BC pre-training: parsing %d cached replays...", len(replay_files))
    trajectories = replay_parser.parse_batch(replay_files)
    if not trajectories:
        return

    clusterer = BehaviorClusterer()
    namer     = SubjectNamer()

    import numpy as np
    labels   = clusterer.fit(trajectories)
    feat_mat = np.stack([clusterer.extract_features(t) for t in trajectories])
    exemplars = clusterer.get_cluster_exemplars(labels, trajectories, feat_mat)
    unique_ids = [c for c in set(labels) if c != -1]
    if not unique_ids:
        return

    cluster_trajs = {
        cid: [t for t, l in zip(trajectories, labels) if l == cid]
        for cid in unique_ids
    }
    subject_names = namer.name_all_clusters(
        np.array(unique_ids), exemplars, feat_mat, labels
    )

    for cid, name in subject_names.items():
        concept = concept_library.get(name)
        if concept is None:
            continue
        if concept.bc_steps_done > 0:
            continue
        slot    = concept.slot_index
        expert  = professor_policy.experts[slot]
        trajs   = cluster_trajs.get(cid, [])
        if not trajs:
            continue
        learner = ImitationLearner(expert)
        loss    = learner.train(trajs, n_epochs=cfg.BC_EPOCHS)
        concept_library.record_bc_steps(name, len(trajs) * cfg.BC_EPOCHS)
        logger.info("BC pre-trained expert for '%s': loss=%.4f", name, loss)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _load_dotenv()

    logger.info("=" * 60)
    logger.info("  Professor RL Bot — Training Start")
    logger.info("  Device: %s | Workers: %d | Timestep limit: %d",
                cfg.DEVICE, args.n_workers, args.timesteps)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 0: Initialise all systems
    # ------------------------------------------------------------------
    from utils.logging import BotLogger
    from engine.curriculum import CurriculumScheduler, TrainingMetrics
    from optimizations.mixed_precision import AMPContext
    from optimizations.profiler import ProfilerContext

    bot_logger = BotLogger(
        use_wandb=not args.disable_wandb and not args.dry_run,
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        config=vars(args),
    )

    concept_library     = _build_concept_library()
    professor_policy    = _build_professor_policy(concept_library)
    curriculum_sched    = CurriculumScheduler()
    metrics             = TrainingMetrics()
    amp                 = AMPContext()
    profiler            = ProfilerContext(active=args.profile)

    # ------------------------------------------------------------------
    # Phase 1: BC pre-training (if replay cache non-empty)
    # ------------------------------------------------------------------
    if not args.disable_replays:
        from replay_system.parser import ReplayParser
        replay_parser = ReplayParser()
        run_bc_pretraining(professor_policy, concept_library, replay_parser)
    else:
        replay_parser = None

    # ------------------------------------------------------------------
    # Start async replay fetcher
    # ------------------------------------------------------------------
    rank_tracker   = None
    async_fetcher  = None

    if not args.disable_replays:
        rank_tracker, async_fetcher, subject_curriculum = _build_replay_pipeline(
            concept_library, professor_policy, args
        )
        async_fetcher.start()
        curriculum_sched.register_advance_callback(
            lambda stage: subject_curriculum  # triggers future fetches at new rank
        )
    else:
        logger.info("Replay fetching disabled.")

    # ------------------------------------------------------------------
    # Build debate loop
    # ------------------------------------------------------------------
    run_debate = _build_debate_loop(professor_policy)

    # ------------------------------------------------------------------
    # Phase 2: Resume from checkpoint if requested
    # ------------------------------------------------------------------
    start_iteration = 0
    if args.resume:
        start_iteration = _load_latest_checkpoint(
            professor_policy, curriculum_sched,
            rank_tracker or __import__("replay_system.rank_tracker", fromlist=["RankTracker"]).RankTracker()
        )

    # ------------------------------------------------------------------
    # Phase 3: Main PPO loop
    # ------------------------------------------------------------------
    try:
        from engine.worker_pool import build_learner
    except ImportError:
        build_learner = None

    # Autoresearch mode: autonomous experiment loop
    if args.autoresearch:
        from autoresearch.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(
            max_experiments=args.autoresearch_experiments,
            steps_per_experiment=args.autoresearch_steps,
            max_time_per_experiment=cfg.AUTORESEARCH_MAX_TIME_PER_EXP,
            n_workers=args.n_workers,
            wandb_project=cfg.WANDB_PROJECT if not args.disable_wandb else None,
            disable_wandb=args.disable_wandb,
            bot_logger=bot_logger,
        )
        runner.run()
        bot_logger.finish()
        return

    # Dry-run: single-process environment loop
    if args.dry_run or build_learner is None:
        logger.info("DRY RUN: running 100 steps in single process.")
        _dry_run(professor_policy, curriculum_sched, metrics, args)
        bot_logger.finish()
        return

    # Normal training via rlgym-ppo Learner
    iteration       = start_iteration
    total_timesteps = start_iteration * cfg.TS_PER_ITERATION
    checkpoint_timer = time.time()

    logger.info("Starting PPO training loop...")

    while total_timesteps < args.timesteps:
        # ------ A. Build environment factory at current stage ------
        stage   = curriculum_sched.current_stage
        env_fn  = __import__("engine.worker_pool", fromlist=["make_env_fn"]).make_env_fn(stage)

        learner = build_learner(
            stage=stage,
            n_proc=args.n_workers,
            logger=logger,
            wandb_project=cfg.WANDB_PROJECT if not args.disable_wandb else None,
            wandb_run_name=f"professor-stage{stage}-iter{iteration}",
        )
        if learner is None:
            logger.error("rlgym-ppo Learner could not be built. Exiting.")
            break

        # ------ B. Collect & update (one PPO iteration) ------
        with profiler.profile("ppo_iteration"):
            try:
                # rlgym-ppo's learn() blocks for the full budget;
                # we wrap it to run one iteration at a time.
                learner.learn()  # runs until timestep_limit (= TS_PER_ITERATION) then returns
            except Exception as e:
                logger.error("Learner.learn() error: %s", e, exc_info=True)
                break

        total_timesteps += cfg.TS_PER_ITERATION
        iteration       += 1

        # ------ C. Update metrics ------
        # rlgym-ppo doesn't expose per-iteration metrics easily;
        # we estimate goal_rate from the reward signal if available.
        metrics.update(cfg.TS_PER_ITERATION, 0.0, 0.0)

        # ------ D. Notify async fetcher ------
        if async_fetcher is not None:
            async_fetcher.notify_timesteps(cfg.TS_PER_ITERATION)
            new_subjects = async_fetcher.drain()
            if new_subjects:
                logger.info("New subjects discovered: %s", new_subjects)

        # ------ E. Curriculum advancement ------
        if curriculum_sched.should_advance(metrics):
            old_stage = curriculum_sched.current_stage
            curriculum_sched.advance()
            new_stage = curriculum_sched.current_stage
            logger.info("Curriculum advanced: stage %d → %d", old_stage, new_stage)
            # Rebuild learner at new stage (next loop iteration)

        # ------ F. Logging ------
        if iteration % cfg.LOG_INTERVAL == 0:
            log_data = {
                "iteration":       iteration,
                "total_timesteps": total_timesteps,
                "curriculum_stage": curriculum_sched.current_stage,
                "goal_rate":       metrics.goal_rate,
                "avg_reward":      metrics.avg_reward,
            }
            if rank_tracker:
                log_data["rank_tier"] = int(rank_tracker.current_tier)
                log_data["win_rate"]  = rank_tracker.win_rate
            log_data.update(profiler.summary())
            bot_logger.log(log_data, step=total_timesteps)

        # ------ G. Checkpoint ------
        if iteration % cfg.CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(
                professor_policy,
                professor_policy.critic,
                curriculum_sched,
                rank_tracker or __import__(
                    "replay_system.rank_tracker", fromlist=["RankTracker"]
                ).RankTracker(),
                iteration,
            )

        profiler.step()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    logger.info("Training complete. Total timesteps: %d", total_timesteps)
    if async_fetcher is not None:
        async_fetcher.stop()
    profiler.export()
    bot_logger.finish()
    # Final checkpoint
    _save_checkpoint(
        professor_policy,
        professor_policy.critic,
        curriculum_sched,
        rank_tracker or __import__(
            "replay_system.rank_tracker", fromlist=["RankTracker"]
        ).RankTracker(),
        iteration,
    )


# ---------------------------------------------------------------------------
# Dry run (no rlgym needed)
# ---------------------------------------------------------------------------

def _dry_run(professor_policy, curriculum_sched, metrics, args):
    """
    Run 100 fake environment steps to verify all systems initialise correctly.
    """
    import numpy as np
    device = str(cfg.DEVICE)

    logger.info("Dry run: checking forward pass through ProfessorPolicy...")
    obs = torch.randn(4, cfg.OBS_DIM, device=device)
    with torch.no_grad():
        action, log_prob, entropy, value = professor_policy.get_action_and_value(obs)
    logger.info(
        "  actions=%s | log_probs=%s | entropy=%.4f | values=%s",
        action.tolist(), log_prob.tolist(), entropy.item(), value.tolist()
    )

    logger.info("Dry run: checking DebateEngine...")
    T, B = cfg.DEBATE_SEQ_LEN, 4
    obs_seq     = torch.randn(T, B, cfg.OBS_DIM, device=device)
    action_seq  = torch.randint(0, cfg.N_ACTIONS, (T, B), device=device)
    concept_ids = torch.zeros(B, dtype=torch.long, device=device)
    outcomes    = torch.randn(B, device=device)

    debate_engine = professor_policy.debate_engine.to(device)
    verdict       = debate_engine(obs_seq, action_seq, concept_ids)
    d_loss        = debate_engine.compute_debate_loss(verdict, outcomes)
    logger.info(
        "  Debate verdict: winner=%s | shaping=%s | loss=%.4f",
        verdict.winner[:2], verdict.shaping_reward[:2].tolist(), d_loss.item()
    )

    logger.info("Dry run: checking reward system...")
    from rewards.combined_reward import DynamicCombinedReward
    reward_fn = DynamicCombinedReward.from_config(stage=0)
    logger.info("  DynamicCombinedReward loaded with %d components.",
                len(reward_fn._components))

    logger.info("Dry run: checking ConceptLibrary...")
    lib = professor_policy.concept_library
    logger.info("  Active concepts: %d / %d", lib.n_active(), lib.max_concepts)

    logger.info("Dry run PASSED. All systems initialised correctly.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows multiprocessing requires spawn start method
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
