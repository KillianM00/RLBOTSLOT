"""
Data models for the autoresearch experiment system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Hypothesis:
    """A single proposed config mutation."""
    experiment_id: str                          # e.g. "exp_0042"
    description: str                            # human-readable summary
    mutations: Dict[str, Any]                   # {"POLICY_LR": 1.5e-4, "REWARD_WEIGHTS.0.TouchReward": 1.5}
    strategy: str                               # "perturbation" | "reward_shuffle" | "schedule_search" | ...
    parent_experiment_id: Optional[str] = None  # which experiment this derives from
    requires_fresh_model: bool = False          # True if mutation changes network shape


@dataclass
class ExperimentResult:
    """Outcome of a single autoresearch experiment."""
    hypothesis: Hypothesis
    baseline_metrics: Dict[str, float]
    experiment_metrics: Dict[str, float]
    accepted: bool
    duration_seconds: float
    reason: str = ""                            # human-readable accept/reject reason
    error: Optional[str] = None                 # set if experiment crashed

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "experiment_id": self.hypothesis.experiment_id,
            "description": self.hypothesis.description,
            "strategy": self.hypothesis.strategy,
            "mutations": self.hypothesis.mutations,
            "parent_experiment_id": self.hypothesis.parent_experiment_id,
            "requires_fresh_model": self.hypothesis.requires_fresh_model,
            "baseline_metrics": self.baseline_metrics,
            "experiment_metrics": self.experiment_metrics,
            "accepted": self.accepted,
            "duration_seconds": round(self.duration_seconds, 2),
            "reason": self.reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentResult:
        """Deserialize from a JSON dict."""
        hyp = Hypothesis(
            experiment_id=d["experiment_id"],
            description=d["description"],
            mutations=d["mutations"],
            strategy=d["strategy"],
            parent_experiment_id=d.get("parent_experiment_id"),
            requires_fresh_model=d.get("requires_fresh_model", False),
        )
        return cls(
            hypothesis=hyp,
            baseline_metrics=d["baseline_metrics"],
            experiment_metrics=d["experiment_metrics"],
            accepted=d["accepted"],
            duration_seconds=d["duration_seconds"],
            reason=d.get("reason", ""),
            error=d.get("error"),
        )
