# Research Program: RLBOTSLOT

## Focus
- Stage 0-1 reward tuning for fast kickoff and possession learning
- Learning rate exploration (is 1e-4 optimal or should we anneal?)
- Entropy coefficient sweep (balance exploration vs exploitation)
- Reward weight ratios (BallProximityReward vs TouchReward vs GoalScoredReward)
- PPO epochs and minibatch size tuning

## Frozen
- N_ACTIONS
- OBS_DIM
- MAX_N_CONCEPTS
- DEVICE
- WANDB_PROJECT
- WANDB_ENTITY

## Bounds
- POLICY_LR: [1e-5, 1e-3]
- CRITIC_LR: [1e-5, 1e-3]
- PPO_ENT_COEF: [0.001, 0.1]
- CLIP_RANGE: [0.1, 0.4]
- GAE_LAMBDA: [0.8, 0.99]
- GAMMA: [0.95, 0.999]
- Reward weights: [0.0, 10.0]
- GOAL_REWARD: [1.0, 50.0]
- GATING_NOISE_STD: [0.1, 3.0]
- LOAD_BALANCE_COEF: [0.001, 0.1]
- DEBATE_LOSS_COEF: [0.001, 0.05]

## Stop
- max_experiments: 1000
