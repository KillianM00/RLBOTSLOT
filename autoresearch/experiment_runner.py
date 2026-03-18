"""
Core orchestrator for the autoresearch experiment loop.

Coordinates hypothesis generation, config mutation, fixed-budget training,
metric evaluation, and accept/reject decisions.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import config as cfg
from autoresearch import config_patcher
from autoresearch.evaluator import Evaluator
from autoresearch.experiment_log import ExperimentLog
from autoresearch.hypothesis import ExperimentResult, Hypothesis
from autoresearch.hypothesis_generator import HypothesisGenerator
from autoresearch.research_program import ResearchProgram

logger = logging.getLogger("autoresearch.experiment_runner")


class ExperimentRunner:
    """
    Orchestrates the autoresearch loop:
      1. Snapshot baseline (checkpoint + config)
      2. Generate hypothesis (config mutation)
      3. Apply mutation, run fixed-budget training
      4. Evaluate metrics, accept/reject
      5. Log result, repeat
    """

    def __init__(
        self,
        max_experiments: int = 100,
        steps_per_experiment: int = 200_000,
        max_time_per_experiment: int = 600,
        n_workers: int = 4,
        wandb_project: Optional[str] = None,
        disable_wandb: bool = False,
        bot_logger: Any = None,
    ) -> None:
        self.max_experiments = max_experiments
        self.steps_per_experiment = steps_per_experiment
        self.max_time_per_experiment = max_time_per_experiment
        self.n_workers = n_workers
        self.wandb_project = wandb_project if not disable_wandb else None
        self.bot_logger = bot_logger

        # Paths
        self.base_dir = Path(cfg.CHECKPOINT_DIR) / "autoresearch"
        self.baseline_dir = self.base_dir / "baseline"
        self.experiment_dir = self.base_dir / "experiment"

        # Components
        self.log = ExperimentLog(cfg.BASE_DIR / "autoresearch" / "experiments.jsonl")
        self.program = ResearchProgram()
        self.generator = HypothesisGenerator(self.log, self.program)
        self.evaluator = Evaluator(
            improvement_threshold=getattr(cfg, "AUTORESEARCH_IMPROVEMENT_THRESH", 0.01),
        )

    def run(self) -> None:
        """Main autoresearch loop. Blocks until max_experiments reached."""
        logger.info("=" * 60)
        logger.info("  AUTORESEARCH — Starting experiment loop")
        logger.info("  Max experiments: %d | Steps/experiment: %d | Workers: %d",
                     self.max_experiments, self.steps_per_experiment, self.n_workers)
        logger.info("=" * 60)

        # Initialize baseline from current rlgym-ppo checkpoints (if any)
        self._ensure_baseline()

        for i in range(self.max_experiments):
            logger.info("-" * 60)
            logger.info("EXPERIMENT %d / %d", i + 1, self.max_experiments)

            try:
                result = self._run_single_experiment()
            except Exception as e:
                logger.error("Experiment crashed: %s", e, exc_info=True)
                # Create a crash result
                result = ExperimentResult(
                    hypothesis=Hypothesis(
                        experiment_id=self.log.next_experiment_id(),
                        description="CRASHED",
                        mutations={},
                        strategy="crash",
                    ),
                    baseline_metrics={},
                    experiment_metrics={},
                    accepted=False,
                    duration_seconds=0.0,
                    reason="Experiment crashed",
                    error=str(e),
                )
                self.log.append(result)
                continue

            self.log.append(result)

            if result.accepted:
                logger.info(">>> ACCEPTED: %s", result.reason)
            else:
                logger.info(">>> REJECTED: %s", result.reason)

            # Log to wandb if available
            if self.bot_logger:
                self.bot_logger.log({
                    "autoresearch/experiment_id": i,
                    "autoresearch/accepted": int(result.accepted),
                    "autoresearch/baseline_reward": result.baseline_metrics.get("avg_reward", 0),
                    "autoresearch/experiment_reward": result.experiment_metrics.get("avg_reward", 0),
                    "autoresearch/strategy": result.hypothesis.strategy,
                })

        logger.info("=" * 60)
        logger.info("AUTORESEARCH COMPLETE")
        logger.info(self.log.summary())
        logger.info("=" * 60)

    def _run_single_experiment(self) -> ExperimentResult:
        """Run one experiment: mutate -> train -> evaluate -> accept/reject."""
        # 1. Snapshot config
        config_snap = config_patcher.snapshot()

        # 2. Generate hypothesis
        hypothesis = self.generator.generate()
        logger.info("Hypothesis: %s", hypothesis.description)
        logger.info("Mutations: %s", hypothesis.mutations)

        # 3. Collect baseline metrics
        baseline_metrics = self._read_baseline_metrics()

        # 4. Apply mutations
        config_patcher.apply_mutations(hypothesis.mutations)

        # 5. Prepare experiment checkpoint
        self._prepare_experiment_checkpoint(hypothesis.requires_fresh_model)

        # 6. Run training
        start_time = time.time()
        experiment_metrics: Dict[str, float] = {}

        try:
            experiment_metrics = self._run_training(hypothesis)
        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            config_patcher.restore(config_snap)
            return ExperimentResult(
                hypothesis=hypothesis,
                baseline_metrics=baseline_metrics,
                experiment_metrics={},
                accepted=False,
                duration_seconds=time.time() - start_time,
                reason="Training failed",
                error=str(e),
            )

        duration = time.time() - start_time

        # 7. Evaluate
        accepted, reason = self.evaluator.evaluate(baseline_metrics, experiment_metrics)

        # 8. Accept or reject
        if accepted:
            # Update baseline with experiment checkpoint
            self._promote_experiment_to_baseline()
            # Config stays as mutated
        else:
            # Restore config and keep baseline unchanged
            config_patcher.restore(config_snap)

        return ExperimentResult(
            hypothesis=hypothesis,
            baseline_metrics=baseline_metrics,
            experiment_metrics=experiment_metrics,
            accepted=accepted,
            duration_seconds=duration,
            reason=reason,
        )

    def _run_training(self, hypothesis: Hypothesis) -> Dict[str, float]:
        """Run fixed-budget training and return collected metrics."""
        from engine.worker_pool import build_learner

        iterations = max(1, self.steps_per_experiment // cfg.TS_PER_ITERATION)
        exp_ckpt = str(self.experiment_dir)

        total_reward = 0.0
        reward_count = 0
        steps_per_sec = 0.0
        start = time.time()

        for i in range(iterations):
            if time.time() - start > self.max_time_per_experiment:
                logger.warning("Wall clock limit reached after %d/%d iterations", i, iterations)
                break

            learner = build_learner(
                stage=0,  # autoresearch focuses on current stage
                n_proc=self.n_workers,
                logger=logger,
                checkpoint_folder=exp_ckpt,
                wandb_project=self.wandb_project,
                wandb_run_name=f"autoresearch-{hypothesis.experiment_id}-iter{i}",
            )
            if learner is None:
                raise RuntimeError("build_learner() returned None — rlgym-ppo not available")

            try:
                learner.learn()
                # Extract metrics from learner
                if hasattr(learner, "agent") and learner.agent.average_reward is not None:
                    total_reward += learner.agent.average_reward
                    reward_count += 1
            finally:
                # Ensure worker processes are cleaned up
                if hasattr(learner, "agent") and hasattr(learner.agent, "cleanup"):
                    try:
                        learner.agent.cleanup()
                    except Exception:
                        pass

            elapsed = time.time() - start
            if elapsed > 0:
                steps_per_sec = (i + 1) * cfg.TS_PER_ITERATION / elapsed

        # Also try reading from checkpoint's BOOK_KEEPING_VARS.json
        checkpoint_metrics = self._read_checkpoint_metrics(self.experiment_dir)

        avg_reward = total_reward / max(reward_count, 1)
        if "policy_average_reward" in checkpoint_metrics:
            avg_reward = checkpoint_metrics["policy_average_reward"]

        return {
            "avg_reward": avg_reward,
            "goal_rate": 0.0,  # TrainingMetrics not accessible from learner
            "steps_per_second": steps_per_sec,
            "iterations_completed": reward_count,
        }

    def _ensure_baseline(self) -> None:
        """Initialize baseline checkpoint dir if it doesn't exist."""
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # If rlgym_ppo has existing checkpoints, copy the latest as baseline
        rlgym_ckpt = Path(cfg.CHECKPOINT_DIR) / "rlgym_ppo"
        if rlgym_ckpt.exists():
            subfolders = []
            for item in rlgym_ckpt.iterdir():
                if item.is_dir():
                    try:
                        subfolders.append((int(item.name), item))
                    except ValueError:
                        pass
            if subfolders:
                subfolders.sort()
                latest = subfolders[-1][1]
                # Copy latest checkpoint as baseline
                target = self.baseline_dir / latest.name
                if not target.exists():
                    shutil.copytree(latest, target)
                    logger.info("Initialized baseline from rlgym-ppo checkpoint: %s", latest.name)

    def _prepare_experiment_checkpoint(self, fresh: bool) -> None:
        """Copy baseline to experiment dir (or clear for fresh start)."""
        # Clear experiment dir
        if self.experiment_dir.exists():
            shutil.rmtree(self.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if fresh:
            logger.info("Fresh model requested — experiment starts from scratch")
            return

        # Copy baseline contents to experiment dir
        if self.baseline_dir.exists():
            for item in self.baseline_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, self.experiment_dir / item.name)
                else:
                    shutil.copy2(item, self.experiment_dir / item.name)
            logger.debug("Copied baseline to experiment dir")

    def _promote_experiment_to_baseline(self) -> None:
        """Replace baseline with experiment checkpoint (on accept)."""
        if self.baseline_dir.exists():
            shutil.rmtree(self.baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        for item in self.experiment_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, self.baseline_dir / item.name)
            else:
                shutil.copy2(item, self.baseline_dir / item.name)
        logger.info("Promoted experiment checkpoint to baseline")

    def _read_baseline_metrics(self) -> Dict[str, float]:
        """Read metrics from baseline checkpoint."""
        metrics = self._read_checkpoint_metrics(self.baseline_dir)
        return {
            "avg_reward": metrics.get("policy_average_reward", 0.0),
            "goal_rate": 0.0,
            "steps_per_second": 0.0,
        }

    @staticmethod
    def _read_checkpoint_metrics(ckpt_dir: Path) -> Dict[str, Any]:
        """Read BOOK_KEEPING_VARS.json from the latest checkpoint subfolder."""
        if not ckpt_dir.exists():
            return {}

        # Find latest numbered subfolder
        subfolders = []
        for item in ckpt_dir.iterdir():
            if item.is_dir():
                try:
                    subfolders.append((int(item.name), item))
                except ValueError:
                    pass
        if not subfolders:
            return {}

        subfolders.sort()
        latest = subfolders[-1][1]
        bk_path = latest / "BOOK_KEEPING_VARS.json"
        if not bk_path.exists():
            return {}

        try:
            with open(bk_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read checkpoint metrics: %s", e)
            return {}
