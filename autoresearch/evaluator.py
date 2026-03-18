"""
Evaluator: compares experiment metrics against baseline to accept or reject.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

logger = logging.getLogger("autoresearch.evaluator")


class Evaluator:
    """
    Compares experiment metrics to baseline metrics using a composite score.

    Score = avg_reward * reward_weight
          + goal_rate  * goal_weight
          + steps_per_second * throughput_weight

    Accepts if experiment_score > baseline_score * (1 + improvement_threshold).
    """

    def __init__(
        self,
        improvement_threshold: float = 0.01,
        reward_weight: float = 1.0,
        goal_weight: float = 2.0,
        throughput_weight: float = 0.0,
    ) -> None:
        self.improvement_threshold = improvement_threshold
        self.reward_weight = reward_weight
        self.goal_weight = goal_weight
        self.throughput_weight = throughput_weight

    def _score(self, metrics: Dict[str, float]) -> float:
        """Compute composite score from a metrics dict."""
        return (
            metrics.get("avg_reward", 0.0) * self.reward_weight
            + metrics.get("goal_rate", 0.0) * self.goal_weight
            + metrics.get("steps_per_second", 0.0) * self.throughput_weight
        )

    def evaluate(
        self,
        baseline_metrics: Dict[str, float],
        experiment_metrics: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Compare experiment to baseline.

        Returns:
            (accepted, reason) tuple.
        """
        baseline_score = self._score(baseline_metrics)
        experiment_score = self._score(experiment_metrics)

        # Handle edge case: both scores are zero (early training)
        if baseline_score == 0.0 and experiment_score == 0.0:
            # Accept if avg_reward improved at all
            br = experiment_metrics.get("avg_reward", 0.0)
            bb = baseline_metrics.get("avg_reward", 0.0)
            if br > bb:
                return True, f"Both scores zero but avg_reward improved ({bb:.4f} -> {br:.4f})"
            return False, f"Both scores zero, no improvement ({bb:.4f} -> {br:.4f})"

        if baseline_score == 0.0:
            # Any positive experiment score is an improvement
            if experiment_score > 0:
                return True, f"Baseline score 0, experiment score {experiment_score:.4f}"
            return False, f"Baseline score 0, experiment score {experiment_score:.4f} (not positive)"

        improvement = (experiment_score - baseline_score) / abs(baseline_score)

        if improvement >= self.improvement_threshold:
            return True, (
                f"Score improved by {improvement:.2%} "
                f"({baseline_score:.4f} -> {experiment_score:.4f})"
            )

        if improvement > -self.improvement_threshold:
            # Within noise tolerance — check if any individual metric improved significantly
            reward_improvement = (
                experiment_metrics.get("avg_reward", 0.0)
                - baseline_metrics.get("avg_reward", 0.0)
            )
            if reward_improvement > 0:
                return True, (
                    f"Score within noise ({improvement:.2%}) "
                    f"but avg_reward improved by {reward_improvement:.4f}"
                )

        return False, (
            f"Score declined by {improvement:.2%} "
            f"({baseline_score:.4f} -> {experiment_score:.4f})"
        )
