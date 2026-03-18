"""
Algorithmic hypothesis generation for autoresearch.

Five strategies, selected by weighted random sampling based on historical
success rates (lightweight population-based training approach).
No LLM calls — purely algorithmic.
"""
from __future__ import annotations

import math
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

import config as cfg
from autoresearch.hypothesis import Hypothesis
from autoresearch.experiment_log import ExperimentLog
from autoresearch.research_program import ResearchProgram
from autoresearch import config_patcher

logger = logging.getLogger("autoresearch.hypothesis_generator")

# Preset architecture choices for EXPERT_HIDDEN
_HIDDEN_PRESETS: List[tuple] = [
    (256, 128),
    (512, 256),
    (512, 256, 128),
    (256, 256),
    (384, 192),
]


class HypothesisGenerator:
    """
    Generates config mutations using five algorithmic strategies:

    1. RandomPerturbation — +-10-50% on a random numeric param
    2. RewardWeightShuffle — adjust one reward component weight
    3. ScheduleSearch — LR / entropy coefficient combos
    4. CurriculumThresholdAdjust — move stage thresholds +-20%
    5. ArchitectureTweak — change EXPERT_HIDDEN to a different preset
    """

    # Strategy names and base weights (updated by history)
    STRATEGIES: List[str] = [
        "perturbation",
        "reward_shuffle",
        "schedule_search",
        "curriculum_adjust",
        "architecture_tweak",
    ]
    BASE_WEIGHTS: List[float] = [3.0, 4.0, 3.0, 1.0, 0.5]

    def __init__(
        self,
        experiment_log: ExperimentLog,
        research_program: ResearchProgram,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.log = experiment_log
        self.program = research_program
        self.rng = random.Random(rng_seed)

    def generate(self) -> Hypothesis:
        """Generate the next hypothesis using a weighted-random strategy."""
        weights = self._compute_strategy_weights()
        strategy = self.rng.choices(self.STRATEGIES, weights=weights, k=1)[0]

        experiment_id = self.log.next_experiment_id()
        accepted = self.log.get_accepted()
        parent_id = accepted[-1].hypothesis.experiment_id if accepted else None

        dispatch = {
            "perturbation": self._random_perturbation,
            "reward_shuffle": self._reward_weight_shuffle,
            "schedule_search": self._schedule_search,
            "curriculum_adjust": self._curriculum_threshold_adjust,
            "architecture_tweak": self._architecture_tweak,
        }

        mutations, description = dispatch[strategy]()
        requires_fresh = config_patcher.mutations_require_fresh_model(mutations)

        return Hypothesis(
            experiment_id=experiment_id,
            description=description,
            mutations=mutations,
            strategy=strategy,
            parent_experiment_id=parent_id,
            requires_fresh_model=requires_fresh,
        )

    def _compute_strategy_weights(self) -> List[float]:
        """Adjust strategy weights by historical accept rate."""
        stats = self.log.get_stats_by_dimension()
        weights = list(self.BASE_WEIGHTS)

        # Map strategies to the param dimensions they typically mutate
        strategy_params = {
            "perturbation": ["POLICY_LR", "CRITIC_LR", "PPO_ENT_COEF", "CLIP_RANGE",
                             "GAE_LAMBDA", "GAMMA"],
            "reward_shuffle": ["REWARD_WEIGHTS"],
            "schedule_search": ["POLICY_LR", "CRITIC_LR", "PPO_ENT_COEF"],
            "curriculum_adjust": ["CURRICULUM_THRESHOLDS"],
            "architecture_tweak": ["EXPERT_HIDDEN"],
        }

        for i, strategy in enumerate(self.STRATEGIES):
            related = strategy_params.get(strategy, [])
            rates = [
                stats[p]["accept_rate"]
                for p in related
                if p in stats and stats[p]["n_tried"] >= 2
            ]
            if rates:
                avg_rate = sum(rates) / len(rates)
                # Boost weight proportional to success: base * (1 + accept_rate)
                weights[i] *= (1.0 + avg_rate)

        return weights

    # ----- Strategy implementations -----

    def _random_perturbation(self) -> Tuple[Dict[str, Any], str]:
        """Pick a random scalar param and perturb by +-10-50%."""
        candidates = [
            "POLICY_LR", "CRITIC_LR", "PPO_ENT_COEF", "CLIP_RANGE",
            "GAE_LAMBDA", "GAMMA", "MAX_GRAD_NORM", "GATING_NOISE_STD",
            "LOAD_BALANCE_COEF", "DEBATE_LOSS_COEF",
            "GOAL_REWARD", "TOUCH_AERIAL_MULTIPLIER",
        ]
        # Remove frozen params
        candidates = [c for c in candidates if c not in self.program.frozen_params]
        if not candidates:
            return {}, "No candidates for perturbation"

        param = self.rng.choice(candidates)
        current = config_patcher.get_current_value(param)

        bounds = self.program.param_bounds.get(param)
        if bounds:
            lo, hi = bounds
            if param in ("POLICY_LR", "CRITIC_LR", "PPO_ENT_COEF",
                         "LOAD_BALANCE_COEF", "DEBATE_LOSS_COEF"):
                # Log-uniform for learning rates
                new_val = math.exp(self.rng.uniform(math.log(lo), math.log(hi)))
            else:
                new_val = self.rng.uniform(lo, hi)
        else:
            # No bounds defined — perturb by +-10-50%
            factor = self.rng.uniform(0.5, 1.5)
            new_val = current * factor

        # Round to reasonable precision
        if isinstance(current, float):
            if abs(new_val) < 0.01:
                new_val = round(new_val, 6)
            else:
                new_val = round(new_val, 5)

        desc = f"Perturb {param}: {current} -> {new_val}"
        return {param: new_val}, desc

    def _reward_weight_shuffle(self) -> Tuple[Dict[str, Any], str]:
        """Pick one reward component at a random stage and adjust its weight."""
        stage = self.rng.randint(0, 4)
        stage_weights = cfg.REWARD_WEIGHTS.get(stage, {})
        if not stage_weights:
            return {}, f"No reward weights for stage {stage}"

        component = self.rng.choice(list(stage_weights.keys()))
        current = stage_weights[component]

        # Adjust by +-20-50%
        factor = self.rng.uniform(0.5, 2.0)
        new_val = round(max(0.0, min(10.0, current * factor)), 2)

        key = f"REWARD_WEIGHTS.{stage}.{component}"
        desc = f"Reward shuffle: stage {stage} {component} {current} -> {new_val}"
        return {key: new_val}, desc

    def _schedule_search(self) -> Tuple[Dict[str, Any], str]:
        """Try a LR / entropy coefficient combo from a log-uniform grid."""
        mutations: Dict[str, Any] = {}
        parts: List[str] = []

        # Always mutate at least one of LR or entropy
        if self.rng.random() < 0.7:
            bounds = self.program.param_bounds.get("POLICY_LR", (1e-5, 1e-3))
            lr = math.exp(self.rng.uniform(math.log(bounds[0]), math.log(bounds[1])))
            lr = round(lr, 7)
            mutations["POLICY_LR"] = lr
            mutations["CRITIC_LR"] = lr  # keep critic LR synced
            parts.append(f"LR={lr}")

        if self.rng.random() < 0.5 or not mutations:
            bounds = self.program.param_bounds.get("PPO_ENT_COEF", (0.001, 0.1))
            ent = math.exp(self.rng.uniform(math.log(bounds[0]), math.log(bounds[1])))
            ent = round(ent, 5)
            mutations["PPO_ENT_COEF"] = ent
            parts.append(f"ent={ent}")

        if self.rng.random() < 0.3:
            bounds = self.program.param_bounds.get("CLIP_RANGE", (0.1, 0.4))
            clip = round(self.rng.uniform(bounds[0], bounds[1]), 2)
            mutations["CLIP_RANGE"] = clip
            parts.append(f"clip={clip}")

        desc = f"Schedule search: {', '.join(parts)}"
        return mutations, desc

    def _curriculum_threshold_adjust(self) -> Tuple[Dict[str, Any], str]:
        """Move a stage's goal_rate or min_timesteps threshold by +-20%."""
        stage = self.rng.randint(1, 4)  # skip stage 0 (always 0)
        thresholds = cfg.CURRICULUM_THRESHOLDS.get(stage, {})
        if not thresholds:
            return {}, f"No thresholds for stage {stage}"

        if self.rng.random() < 0.5:
            # Adjust goal_rate
            current = thresholds["goal_rate"]
            factor = self.rng.uniform(0.8, 1.2)
            new_val = round(max(0.05, min(0.9, current * factor)), 3)
            key = f"CURRICULUM_THRESHOLDS.{stage}.goal_rate"
            desc = f"Curriculum: stage {stage} goal_rate {current} -> {new_val}"
        else:
            # Adjust min_timesteps
            current = thresholds["min_timesteps"]
            factor = self.rng.uniform(0.8, 1.2)
            new_val = int(current * factor)
            key = f"CURRICULUM_THRESHOLDS.{stage}.min_timesteps"
            desc = f"Curriculum: stage {stage} min_timesteps {current} -> {new_val}"

        return {key: new_val}, desc

    def _architecture_tweak(self) -> Tuple[Dict[str, Any], str]:
        """Change EXPERT_HIDDEN to a different preset. Requires fresh model."""
        current = cfg.EXPERT_HIDDEN
        choices = [p for p in _HIDDEN_PRESETS if p != current]
        if not choices:
            return {}, "No alternative architectures"

        new_hidden = self.rng.choice(choices)
        desc = f"Architecture: EXPERT_HIDDEN {current} -> {new_hidden}"
        return {"EXPERT_HIDDEN": new_hidden}, desc
