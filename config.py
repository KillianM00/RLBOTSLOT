"""
Central configuration for the RocketLeague AI bot.
All hyperparameters live here — change values here rather than in individual modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR         = Path(__file__).parent
CHECKPOINT_DIR   = BASE_DIR / "models" / "checkpoints"
REPLAY_CACHE_DIR = BASE_DIR / "replay_system" / "replay_cache"
REPLAY_DB_PATH   = BASE_DIR / "replay_system" / "replay_index.db"
MEMMAP_PATH      = BASE_DIR / "replay_buffer.dat"
LOG_DIR          = BASE_DIR / "logs"
CONCEPT_SAVE_PATH = CHECKPOINT_DIR / "concept_library.pkl"

# ---------------------------------------------------------------------------
# System 1: Hyperaccelerated Training Engine
# ---------------------------------------------------------------------------
N_WORKERS              = 32        # parallel RocketSim worker processes
TICK_SKIP              = 8         # physics ticks per bot action (120 Hz / 8 = 15 Hz)
N_ACTIONS              = 90        # lookup table action space size
TS_PER_ITERATION       = 100_000   # timesteps collected per PPO iteration
TIMESTEP_LIMIT         = 5_000_000_000   # total training budget (~5B)

# Episode parameters
MAX_EPISODE_SECONDS    = 300       # 5 minutes max per episode
NO_TOUCH_TIMEOUT       = 30        # end episode if ball untouched this long (seconds)

# Curriculum stages: 0..4
CURRICULUM_THRESHOLDS  = {
    0: {"min_timesteps": 0,           "goal_rate": 0.0},   # start at stage 0
    1: {"min_timesteps": 5_000_000,   "goal_rate": 0.3},
    2: {"min_timesteps": 50_000_000,  "goal_rate": 0.4},
    3: {"min_timesteps": 200_000_000, "goal_rate": 0.45},
    4: {"min_timesteps": 500_000_000, "goal_rate": 0.5},
}

# ---------------------------------------------------------------------------
# System 2: PPO Hyperparameters
# ---------------------------------------------------------------------------
PPO_EPOCHS             = 2
PPO_ENT_COEF           = 0.01
POLICY_LR              = 1e-4
CRITIC_LR              = 1e-4
GAMMA                  = 0.99
GAE_LAMBDA             = 0.95
CLIP_RANGE             = 0.2
MAX_GRAD_NORM          = 0.5
STANDARDIZE_RETURNS    = True
PPO_BATCH_SIZE         = 100_000
PPO_MINIBATCH_SIZE     = 50_000

# ---------------------------------------------------------------------------
# System 2: Professor / MoE Architecture
# ---------------------------------------------------------------------------
INITIAL_N_CONCEPTS     = 8        # expert slots active at start
MAX_N_CONCEPTS         = 64       # pre-allocated; unused slots are masked off
K_EXPERTS              = 2        # top-k routing in GatingNetwork
EXPERT_HIDDEN          = (512, 256)
GATING_HIDDEN          = (256,)
EMBED_DIM              = 128      # concept embedding dimension
OBS_DIM                = 98       # DefaultObs(1v1)=92 + 6 extra derived features

# Noisy top-k gating
GATING_NOISE_STD       = 1.0      # noise weight for noisy gating

# Aux loss coefficients
LOAD_BALANCE_COEF      = 0.01     # lambda_lb
DEBATE_LOSS_COEF       = 0.005    # lambda_d
BC_REGULARIZE_COEF     = 0.1      # lambda_bc during imitation regularization period

# Debate engine
DEBATE_GRU_HIDDEN      = 256
DEBATE_SEQ_LEN         = 30       # trajectory window length for debate (frames)
MAX_DEBATES_PER_ITER   = 1_000    # cap on async debate computations per PPO step
MIN_CONCEPT_USAGE      = 10       # min usage before a concept participates in debates

# ---------------------------------------------------------------------------
# System 3: Reward Weights by Curriculum Stage
# ---------------------------------------------------------------------------
# Each stage's dict is passed to DynamicCombinedReward.update_weights()
REWARD_WEIGHTS = {
    0: {
        "BallProximityReward":      0.1,   # dense shaping — fires every step
        "TouchReward":              1.0,
        "GoalScoredReward":         5.0,
        "AntiChaseReward":          0.5,
        "CurriculumProgressReward": 1.0,
    },
    1: {
        "TouchReward":              0.5,
        "GoalScoredReward":         5.0,
        "BallControlReward":        1.0,
        "ShotReward":               1.0,
        "AntiChaseReward":          0.8,
        "CurriculumProgressReward": 0.5,
    },
    2: {
        "TouchReward":              0.3,
        "GoalScoredReward":         5.0,
        "BallControlReward":        0.8,
        "ShotReward":               1.0,
        "SaveReward":               1.0,
        "BoostEfficiencyReward":    0.5,
        "SpeedReward":              0.3,
        "PositioningReward":        0.5,
        "AntiChaseReward":          0.8,
        "ConceptDebateReward":      0.3,
        "CurriculumProgressReward": 0.3,
    },
    3: {
        "TouchReward":              0.2,
        "GoalScoredReward":         5.0,
        "BallControlReward":        0.6,
        "ShotReward":               1.0,
        "SaveReward":               1.0,
        "BoostEfficiencyReward":    0.5,
        "SpeedReward":              0.3,
        "PositioningReward":        0.5,
        "AerialReward":             1.0,
        "FlickReward":              0.8,
        "DemoReward":               0.3,
        "AntiChaseReward":          0.8,
        "ConceptDebateReward":      0.5,
        "CurriculumProgressReward": 0.2,
    },
    4: {
        "TouchReward":              0.1,
        "GoalScoredReward":         5.0,
        "BallControlReward":        0.5,
        "ShotReward":               1.0,
        "SaveReward":               1.0,
        "BoostEfficiencyReward":    0.6,
        "SpeedReward":              0.4,
        "PositioningReward":        0.7,
        "AerialReward":             1.0,
        "FlickReward":              1.0,
        "DemoReward":               0.4,
        "AntiChaseReward":          1.0,
        "ConceptDebateReward":      0.7,
        "CurriculumProgressReward": 0.0,
    },
}

# Individual reward constants
GOAL_REWARD                    = 10.0
CONCEDE_REWARD                 = -10.0
TOUCH_AERIAL_MULTIPLIER        = 3.0
FLICK_VELOCITY_THRESHOLD       = 500.0   # uu/s perpendicular velocity spike
ANTI_CHASE_SAME_TOUCH_PENALTY  = -0.1
ANTI_CHASE_OVERCOMMIT_PENALTY  = -0.2
BOOST_LOW_THRESHOLD            = 0.2     # fraction; penalize collecting when full
DEMO_MIN_DISTANCE              = 2300.0  # uu; only reward demo if ball is close

# ---------------------------------------------------------------------------
# System 4: Replay Watching / Subject Discovery
# ---------------------------------------------------------------------------
BALLCHASING_API_KEY    = ""       # loaded from .env: BALLCHASING_API_KEY
STARTING_RANK          = "silver-1"    # ballchasing rank string
MAX_RANK_GAP           = 2             # tiers above bot rank still usable for imitation
REPLAY_FETCH_INTERVAL  = 1_000_000     # timesteps between replay fetches
REPLAY_BATCH_SIZE      = 200           # replays per fetch
REPLAY_RATE_LIMIT_RPS  = 2.0          # standard tier: 2 req/sec

# Behavior clustering
SEGMENT_LENGTH         = 30            # frames per trajectory window
HDBSCAN_MIN_CLUSTER    = 50
HDBSCAN_MIN_SAMPLES    = 5
CLUSTER_EMBED_DIM      = 64            # feature vector size for clustering

# Imitation learning
BC_EPOCHS              = 5
BC_BATCH_SIZE          = 2048
BC_REGULARIZE_STEPS    = 500_000       # RL steps during which BC loss is mixed in

# Concept pruning
CONCEPT_MIN_WIN_RATE   = 0.10          # prune concepts below this debate win rate

# Subject name prefixes (assigned heuristically based on dominant features)
SUBJECT_PREFIXES = [
    "AerialRecovery", "FlickSequence", "ShadowDefense",
    "BoostDeny", "WallRead", "ResetChain", "CeilingShot",
    "BallchaseBreak", "RotationCover", "MustyFlick",
    "PinchSetup", "NeutralRead", "BoostStarve",
    "DoubleTap", "RollingGroundShot", "DirectionChange",
]

# ---------------------------------------------------------------------------
# System 5: Efficiency Optimizations
# ---------------------------------------------------------------------------
COMPILE_MODE           = "reduce-overhead"   # torch.compile mode
USE_AMP                = True                # mixed precision (BF16/FP16 auto)
REPLAY_BUFFER_CAPACITY = 1_000_000
MIN_INFERENCE_BATCH    = 32                  # min obs to batch before GPU forward pass

# ---------------------------------------------------------------------------
# Device detection (auto-selects CUDA > DirectML > CPU)
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
WANDB_PROJECT          = "rlbot-professor"
WANDB_ENTITY           = ""        # your wandb username/org, or leave blank
LOG_INTERVAL           = 10        # log every N PPO iterations
CHECKPOINT_INTERVAL    = 50        # save checkpoint every N iterations

# ---------------------------------------------------------------------------
# System 6: Autoresearch
# ---------------------------------------------------------------------------
AUTORESEARCH_STEPS_PER_EXPERIMENT = 100_000   # 1 PPO iteration per experiment (~2-3 min)
AUTORESEARCH_MAX_TIME_PER_EXP    = 300        # 5 min wall clock cap
AUTORESEARCH_IMPROVEMENT_THRESH  = 0.01       # 1% minimum improvement to accept
AUTORESEARCH_MAX_EXPERIMENTS     = 1000        # max experiments per session
