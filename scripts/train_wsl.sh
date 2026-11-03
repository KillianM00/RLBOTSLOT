#!/bin/bash
set -euo pipefail
RLBOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RLBOT_VENV="${RLBOT_VENV:-$HOME/rlbot-venv}"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
source "$RLBOT_VENV/bin/activate"
cd "$RLBOT_ROOT"
python train.py "$@"
