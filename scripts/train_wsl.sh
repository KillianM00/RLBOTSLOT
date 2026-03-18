#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
source ~/rlbot-venv/bin/activate
cd $HOME/RLBOTSLOT
python train.py "$@"
