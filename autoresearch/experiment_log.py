"""
Append-only JSONL experiment log for autoresearch.

Each line in experiments.jsonl is one ExperimentResult serialized as JSON.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from autoresearch.hypothesis import ExperimentResult

logger = logging.getLogger("autoresearch.experiment_log")


class ExperimentLog:
    """Persistent experiment history backed by a JSONL file."""

    def __init__(self, log_path: str | Path = "autoresearch/experiments.jsonl") -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: ExperimentResult) -> None:
        """Append one experiment result as a JSON line."""
        entry = result.to_dict()
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(
            "Logged %s: %s (%s)",
            result.hypothesis.experiment_id,
            "ACCEPTED" if result.accepted else "REJECTED",
            result.reason,
        )

    def load_all(self) -> List[ExperimentResult]:
        """Load all experiment results from the log."""
        if not self.path.exists():
            return []
        results = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                results.append(ExperimentResult.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping corrupt log line: %s", e)
        return results

    def get_accepted(self) -> List[ExperimentResult]:
        """Return only accepted experiments."""
        return [r for r in self.load_all() if r.accepted]

    def get_stats_by_dimension(self) -> Dict[str, Dict[str, float]]:
        """
        Returns {param_name: {n_tried, n_accepted, accept_rate}} aggregated
        across all experiments that mutated that parameter.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for result in self.load_all():
            for key in result.hypothesis.mutations:
                root = key.split(".")[0]
                if root not in stats:
                    stats[root] = {"n_tried": 0, "n_accepted": 0}
                stats[root]["n_tried"] += 1
                if result.accepted:
                    stats[root]["n_accepted"] += 1
        for v in stats.values():
            v["accept_rate"] = v["n_accepted"] / max(v["n_tried"], 1)
        return stats

    def next_experiment_id(self) -> str:
        """Return the next sequential experiment ID."""
        all_results = self.load_all()
        return f"exp_{len(all_results):04d}"

    def summary(self) -> str:
        """Human-readable summary of all experiments."""
        all_results = self.load_all()
        if not all_results:
            return "No experiments logged yet."
        n_total = len(all_results)
        n_accepted = sum(1 for r in all_results if r.accepted)
        n_rejected = n_total - n_accepted
        best = max(
            all_results,
            key=lambda r: r.experiment_metrics.get("avg_reward", float("-inf")),
        )
        return (
            f"Experiments: {n_total} total, {n_accepted} accepted, {n_rejected} rejected\n"
            f"Accept rate: {n_accepted / n_total:.1%}\n"
            f"Best: {best.hypothesis.experiment_id} "
            f"(avg_reward={best.experiment_metrics.get('avg_reward', 'N/A')}, "
            f"strategy={best.hypothesis.strategy})"
        )
