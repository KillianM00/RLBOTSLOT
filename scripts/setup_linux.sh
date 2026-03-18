#!/bin/bash
# Full setup for Ubuntu 24.04 with AMD RX 6700 XT (ROCm)
# Run: chmod +x scripts/setup_linux.sh && ./scripts/setup_linux.sh
set -e

echo "=== RLBOTSLOT Linux GPU Setup ==="
echo "GPU: AMD RX 6700 XT | ROCm 6.2 | PyTorch ROCm"

# 1. System deps
echo ">>> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv curl wget git cmake gcc g++

# 2. ROCm repo + runtime
echo ">>> Adding ROCm repository..."
sudo mkdir -p /etc/apt/keyrings
curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.2/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/amdgpu.list
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' | sudo tee /etc/apt/preferences.d/rocm-pin-600

echo ">>> Installing ROCm HIP runtime..."
sudo apt-get update -qq
sudo apt-get install -y rocm-hip-runtime

# 3. GFX version override for RX 6700 XT
grep -q HSA_OVERRIDE_GFX_VERSION ~/.bashrc 2>/dev/null || {
    echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
    echo ">>> Added HSA_OVERRIDE_GFX_VERSION to .bashrc"
}
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# 4. Add user to render/video groups (required for GPU access)
sudo usermod -aG render,video $USER

# 5. Python venv + PyTorch ROCm
echo ">>> Creating Python venv..."
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate

echo ">>> Installing PyTorch with ROCm support (~2.5GB download)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

echo ">>> Installing project requirements..."
pip install -r requirements.txt

# 6. Create .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo ">>> Created .env from .env.example — add your API keys!"
fi

# 7. Test GPU
echo ""
echo "=== GPU Detection Test ==="
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA/ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('VRAM:', round(props.total_mem / 1e9, 1), 'GB')
    t = torch.randn(1000, 1000, device='cuda')
    r = t @ t.T
    print('GPU matmul test: PASSED')
else:
    print('WARNING: No GPU detected!')
    print('Make sure you rebooted after installing ROCm')
    print('And that your user is in the render/video groups')
"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Log out and back in (for group membership to take effect)"
echo "  2. source .venv/bin/activate"
echo "  3. Edit .env with your API keys"
echo "  4. python train.py --dry-run"
echo "  5. python train.py --n-workers 8"
