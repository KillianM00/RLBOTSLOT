# Rocket League Bot Research

An experimental reinforcement-learning project combining RocketSim training,
a concept-based professor/debate policy, replay analysis, and configurable
training optimizations.

## Status

Research code, not a released competitive bot. Plans in `docs/PLAN.md` include
proposed optimizations; estimated speedups are targets, not measured results.
No competitive rank, win rate, or hardware speedup is claimed by this README.

## Setup

Use an isolated Python environment compatible with the dependencies in
`requirements.txt`. PyTorch acceleration depends on your hardware and supported
driver/runtime combination; CPU operation is a valid starting point.

```sh
git clone https://github.com/KillianM00/RLBOTSLOT.git
cd RLBOTSLOT
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
```

For replay fetching, copy `.env.example` to `.env` and supply your own
Ballchasing API configuration. Keep credentials and downloaded personal data
out of Git. The initial smoke run can disable replay fetching and telemetry:

```sh
python train.py --dry-run --disable-replays --disable-wandb --n-workers 1
```

## Validation

Run `python -m pytest tests`. For training comparisons, record the commit,
dependency versions, hardware, seeds, worker count, timestep budget, evaluation
opponents, and resulting metrics. Keep evaluation separate from training.

## Source map

| Path | Purpose |
| --- | --- |
| `train.py` | Training entry point and command-line options |
| `config.py` | Experiment and runtime configuration |
| `engine/` | Simulation and worker coordination |
| `professor/` | Concept library and policy components |
| `replay_system/` | Replay ingestion and analysis |
| `optimizations/` | Training optimizations |
| `tests/` | Component tests |
| `docs/PLAN.md` | Development proposals and outstanding work |

## Licensing

No project-wide license has been selected for the original code. Dependencies
and third-party data retain their own terms. Contact the maintainer before reuse.
