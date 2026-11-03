# Plan: GPU Acceleration + Integrated Autoresearch + Quantum Optimization

> Historical research proposal. Performance figures and speedups below are
> unverified targets. Hardware compatibility and installation commands require
> independent validation. See the README for current implementation limitations.

## Context
Two problems: (1) The RX 6700 XT GPU is completely idle — all training runs on CPU at ~1.5-3k steps/sec. (2) Autoresearch and main training are separate systems that don't talk to each other. This plan fixes both: unlock the GPU via WSL2+ROCm for 10-50x speedup, then integrate autoresearch with quantum-inspired search.

---

## Phase 0: GPU Acceleration via WSL2 + ROCm (DO THIS FIRST)

### Why WSL2+ROCm
- DirectML is broken for training (no scatter_, no GRU, no AMP)
- ROCm is AMD's CUDA equivalent — full PyTorch GPU training support
- WSL2 is officially supported by AMD as of ROCm 7.2
- RX 6700 XT (gfx1031) works with `HSA_OVERRIDE_GFX_VERSION=10.3.0`

### Steps
1. **Install Ubuntu 24.04 in WSL2**: `wsl --install -d Ubuntu-24.04`
2. **Inside WSL2**:
   - Install ROCm 7.x: `sudo apt install rocm` (follow AMD's official guide)
   - Set gfx version override: `echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc`
   - Install PyTorch ROCm: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2`
   - Verify: `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
3. **Clone the project** in WSL2, for example into `~/RLBOTSLOT/`.
4. **Create WSL venv**, install all deps including rlgym-ppo
5. **Create `scripts/train_wsl.sh`** — launcher script that sets env vars and runs training
6. **Update `config.py`**: `DEVICE` auto-detection now finds ROCm as `"cuda"` (ROCm uses CUDA API)
7. **Update `engine/worker_pool.py`**: change `device="cpu"` to `device="cuda"` when available
8. **Enable AMP**: ROCm supports mixed precision — `optimizations/mixed_precision.py` works with ROCm
9. **Test**: run `python train.py --dry-run` in WSL2, verify GPU is used

### Expected speedup
- Steps/sec: ~1.5-3k (CPU) → **15-50k (GPU)** depending on batch size
- PPO update: ~30s (CPU) → **1-3s (GPU)**
- Mixed precision (FP16): additional ~1.5x on top of GPU baseline

### Modify: `engine/worker_pool.py`
- Change line 70 from `device="cpu"` to:
  ```python
  import torch
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```

### Modify: `config.py`
- `_detect_device()` already returns "cuda" when ROCm is present (ROCm uses torch.cuda API)
- No changes needed — it will auto-detect

### New file: `scripts/train_wsl.sh`
```bash
#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
cd ~/RLBOTSLOT
source .venv-wsl/bin/activate
python train.py "$@"
```

---

## Phase 1: Unified Training + Autoresearch Loop

### New file: `autoresearch/integrated_trainer.py`

Replaces the either/or branch in `train.py`. Interleaves main PPO training with periodic autoresearch experiments:

```
while total_timesteps < budget:
    # --- Main training: N iterations ---
    for i in range(EXPERIMENT_EVERY_N):   # default 5
        learner = build_learner(stage, n_proc=MAIN_WORKERS)
        learner.learn()                    # 100k steps
        update_metrics, check_curriculum, log, checkpoint

    # --- Autoresearch experiment ---
    hypothesis = quantum_generator.generate()
    config_snap = config_patcher.snapshot()
    config_patcher.apply_mutations(hypothesis.mutations)
    copy main checkpoint -> experiment dir
    exp_learner = build_learner(stage, n_proc=EXPERIMENT_WORKERS, ckpt=experiment_dir)
    exp_learner.learn()                    # 100k steps
    evaluate: accept or reject
    if accepted: copy experiment weights -> main ckpt dir (config stays mutated)
    else: config_patcher.restore(config_snap)
    quantum_generator.update_landscape(hypothesis, score)
```

Key class: `IntegratedTrainer(professor_policy, curriculum_sched, metrics, bot_logger, args)`
- `run()` — the unified loop above
- `_run_experiment()` — one hypothesis/evaluate cycle
- `_hot_apply()` — copy accepted weights into main checkpoint dir

### Modify: `train.py` (lines 365-490)
- Add `--integrated` flag (default True when `--autoresearch` also passed)
- Remove the early `return` at line 379
- When `--integrated`: instantiate `IntegratedTrainer` and call `.run()` instead of the raw while loop
- Legacy `--autoresearch` without `--integrated`: old standalone behavior preserved

### Modify: `engine/worker_pool.py`
- Add Learner caching to avoid respawning worker processes every iteration:
  ```python
  _cached_learner = None
  _cached_key = None

  def build_learner(stage, n_proc, ...):
      key = (stage, n_proc, checkpoint_folder)
      if _cached_learner and _cached_key == key:
          return _cached_learner
      # ... build new ...
  ```
- Add `invalidate_learner_cache()` for when experiment checkpoint replaces main

---

## Phase 2: Quantum-Inspired Hypothesis Generator

### New file: `autoresearch/quantum_hypothesis_generator.py`

Subclasses `HypothesisGenerator`, adds 3 quantum-inspired strategies (8 total):

**Strategy 6: Simulated Quantum Annealing** (`_quantum_anneal`)
- Uses `dwave-neal` SimulatedAnnealingSampler
- Discretize each continuous hyperparam into 8 bits → QUBO binary matrix
- Objective = negative interpolated score from landscape history
- Temperature: `T = 1.0 * 0.995^n_experiments` (starts exploratory, cools to exploitative)
- Allows "tunneling" through local optima that random perturbation can't escape

**Strategy 7: Grover-Inspired Adaptive Sampling** (`_grover_sample`)
- Maintain amplitude vector: `a[param][region]` over 32 bins per parameter
- After accepted experiment: `a[p][r] *= sqrt(2)` (amplify)
- After rejected: `a[p][r] /= sqrt(2)` (attenuate)
- Normalize to probabilities: `P(r) = |a[p][r]|^2 / sum`
- Concentrates search on proven-good regions with O(sqrt(N)) theoretical speedup

**Strategy 8: Quantum Walk on Hyperparameter Graph** (`_quantum_walk`)
- Nodes = visited configurations, edges = configs differing by 1-2 params
- At current node, compute interference from neighbors:
  - Better neighbors → constructive → higher move probability
  - Worse neighbors → destructive → lower move probability
- `P(v->u) = softmax(interference * phase_factor)` where phase decays over time
- Memory bounded to 100 nodes (FIFO prune)

Shared state across strategies:
- `landscape_history: List[(config_vector, score)]` — updated after every experiment
- `temperature: float` — annealing schedule
- `amplitude_map: Dict[param, ndarray]` — Grover amplitudes
- `walk_graph: Dict[node_id, {config, score, neighbors}]` — quantum walk

### Modify: `autoresearch/evaluator.py`
Add `QuantumEvaluator` subclass with Metropolis-Hastings acceptance:
```python
# Even if experiment is worse, accept with probability:
p_accept = exp(-(baseline_score - experiment_score) / temperature)
# Allows tunneling early; becomes strict as T → 0
```

### New dependency: `dwave-neal>=0.6.0` in `requirements.txt`

---

## Phase 3: Speed Optimizations

### Modify: `config.py`
```python
N_WORKERS           = 8        # up from 4 (better CPU utilization)
MAIN_WORKERS        = 6        # for main training in integrated mode
EXPERIMENT_WORKERS  = 2        # for autoresearch experiments
PPO_MINIBATCH_SIZE  = 25_000   # down from 50k (fits L3 cache ~10MB)
EXPERIMENT_EVERY_N  = 5        # experiment every 5 main iterations

# Quantum search
QUANTUM_ANNEAL_INITIAL_TEMP = 1.0
QUANTUM_ANNEAL_COOLING_RATE = 0.995
GROVER_N_REGIONS            = 32
GROVER_AMPLIFY_FACTOR       = 1.414   # sqrt(2)
QUANTUM_WALK_MEMORY         = 100
```

### Modify: `engine/worker_pool.py`
- Learner caching (avoid process respawn overhead ~5-10s per iteration)
- Only rebuild when stage changes or `invalidate_learner_cache()` called

---

## File Summary

| Action | File | Changes |
|--------|------|---------|
| **CREATE** | `autoresearch/quantum_hypothesis_generator.py` | 3 quantum strategies + landscape state (~350 lines) |
| **CREATE** | `autoresearch/integrated_trainer.py` | Unified training+experiment loop (~250 lines) |
| **MODIFY** | `train.py` | Add `--integrated` flag, wire IntegratedTrainer, remove early return |
| **MODIFY** | `engine/worker_pool.py` | Learner caching + invalidation |
| **MODIFY** | `autoresearch/evaluator.py` | Add QuantumEvaluator with annealing acceptance |
| **MODIFY** | `autoresearch/hypothesis_generator.py` | Add `update_landscape()` method to base class |
| **MODIFY** | `config.py` | New defaults: N_WORKERS=8, PPO_MINIBATCH_SIZE=25k, quantum params |
| **MODIFY** | `requirements.txt` | Add `dwave-neal>=0.6.0` |

---

## Implementation Order

1. **Config changes** — `config.py` new defaults + quantum params
2. **Learner caching** — `engine/worker_pool.py`
3. **Quantum hypothesis generator** — new file, subclass of existing
4. **Quantum evaluator** — extend `evaluator.py`
5. **Integrated trainer** — new file, the unified loop
6. **Wire into train.py** — add `--integrated` flag, connect everything
7. **Requirements** — add `dwave-neal`
8. **Test** — dry-run + 3-experiment smoke test

---

## Verification

1. `python train.py --dry-run` — still works (no regression)
2. `python train.py --autoresearch --autoresearch-experiments 3` — legacy mode still works
3. `python train.py --integrated --autoresearch --n-workers 8` — new unified mode:
   - Should run 5 main iterations, then 1 experiment, then 5 main, etc.
   - WandB shows both main training metrics AND autoresearch accept/reject
   - `autoresearch/experiments.jsonl` logs experiments with quantum strategy names
4. Check that accepted experiment configs persist into subsequent main training iterations
5. Check that rejected experiments restore config cleanly
6. After 10+ experiments: verify quantum strategies appear in the log (quantum_anneal, grover_sample, quantum_walk)
