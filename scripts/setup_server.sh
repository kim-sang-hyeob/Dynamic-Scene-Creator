#!/bin/bash
# =============================================================================
# 4DGS Server Setup Script
# Colmap-free Unity→4DGS Pipeline - Server Environment Setup
# =============================================================================
#
# Usage (run as root):
#   chmod +x scripts/setup_server.sh
#   ./scripts/setup_server.sh [--with-vggt]
#
# Options:
#   --with-vggt    Also setup VGGT model (optional, requires more VRAM)
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo " 4DGS Server Setup"
echo "=============================================="

# Parse arguments
INSTALL_VGGT=false
for arg in "$@"; do
    case $arg in
        --with-vggt)
            INSTALL_VGGT=true
            shift
            ;;
    esac
done

# =============================================================================
# 1. System Dependencies
# =============================================================================
echo ""
echo "[1/5] Installing system dependencies..."

apt-get update
apt-get install -y git wget ninja-build

# Graphics/Display libraries (for headless rendering)
apt-get install -y libx11-6 libgl1 libglib2.0-0

echo "[1/5] System dependencies installed."

# =============================================================================
# 2. CUDA 11.8 Setup (for PyTorch compatibility)
# =============================================================================
echo ""
echo "[2/5] Setting up CUDA 11.8..."

# Check if CUDA 11.8 nvcc is already installed
if command -v /usr/local/cuda-11.8/bin/nvcc &> /dev/null; then
    echo "CUDA 11.8 already installed."
else
    echo "Installing CUDA 11.8 toolkit..."

    # Repository setup
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

    # Keyring
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    apt-get update

    # Install CUDA nvcc compiler (11.8 for PyTorch compatibility)
    apt-get install -y cuda-nvcc-11-8
fi

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME=$CUDA_HOME"
echo "[2/5] CUDA 11.8 setup complete."

# =============================================================================
# 3. Python Dependencies
# =============================================================================
echo ""
echo "[3/5] Installing Python dependencies..."

# Core dependencies
pip install --upgrade pip

# Ensure numpy < 2.0 (critical for compatibility)
pip install "numpy<2.0"

# Additional dependencies
pip install websockets

echo "[3/5] Python dependencies installed."

# =============================================================================
# 4. Setup 4DGS Model
# =============================================================================
echo ""
echo "[4/5] Setting up 4DGS model..."

# Navigate to project directory
cd "$(dirname "$0")/.."

# Run 4DGS setup
python manage.py setup --model 4dgs

echo "[4/5] 4DGS setup complete."

# =============================================================================
# 5. (Optional) Setup VGGT Model
# =============================================================================
if [ "$INSTALL_VGGT" = true ]; then
    echo ""
    echo "[5/5] Setting up VGGT model (optional)..."
    python manage.py setup --model vggt
    echo "[5/5] VGGT setup complete."
else
    echo ""
    echo "[5/5] Skipping VGGT setup (use --with-vggt to install)"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================="
echo " Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Process Unity data:"
echo "     python manage.py process-unity <video> <json> <original> --output <name>"
echo ""
echo "  2. Train 4DGS:"
echo "     python manage.py train data/<name>"
echo ""
echo "  3. Render with camera rotation:"
echo "     CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py -m output/4dgs/<name> --skip_train --skip_test"
echo ""
echo "Environment variables set:"
echo "  CUDA_HOME=$CUDA_HOME"
echo ""
echo "To persist CUDA_HOME, add to ~/.bashrc:"
echo "  export CUDA_HOME=/usr/local/cuda-11.8"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo ""
