#!/bin/bash
set -e
RLBOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# === ROCm repo + key ===
echo ">>> Adding ROCm repo..."
sudo mkdir -p /etc/apt/keyrings
curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.2/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/amdgpu.list
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' | sudo tee /etc/apt/preferences.d/rocm-pin-600
echo ">>> Repo added."

# === Install ROCm runtime ===
echo ">>> apt update..."
sudo apt-get update -qq
echo ">>> Installing ROCm HIP runtime (this downloads ~500MB)..."
sudo apt-get install -y rocm-hip-runtime
echo ">>> ROCm installed."

# === GFX override for RX 6700 XT ===
grep -q HSA_OVERRIDE_GFX_VERSION ~/.bashrc 2>/dev/null || echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# === Create venv on NATIVE Linux filesystem (fast!) ===
echo ">>> Creating venv at ~/rlbot-venv (native ext4, not OneDrive)..."
python3 -m venv ~/rlbot-venv
source ~/rlbot-venv/bin/activate

# === Install PyTorch ROCm ===
echo ">>> Installing PyTorch ROCm (~2.5GB download)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# === Install project dependencies from this checkout ===
echo ">>> Installing project requirements..."
pip install -r "$RLBOT_ROOT/requirements.txt"

# === Test GPU ===
echo ">>> Testing GPU detection..."
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA/ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    t = torch.randn(1000, 1000, device='cuda')
    print('GPU tensor test OK, shape:', t.shape)
else:
    print('WARNING: No GPU detected')
"

echo "=== SETUP COMPLETE ==="
echo "To train from Linux or WSL: bash \"$RLBOT_ROOT/scripts/train_wsl.sh\""
