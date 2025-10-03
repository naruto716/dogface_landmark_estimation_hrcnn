#!/bin/bash
# MMPose setup script for SageMaker with GPU
set -e

echo "🚀 Setting up MMPose on SageMaker..."

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "CUDA Version: $CUDA_VERSION"
else
    echo "⚠️  No GPU detected, will use CPU (slower)"
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $PYTHON_VERSION"

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install openmim
echo "📦 Installing openmim..."
pip install -U openmim

# Install mmengine and mmcv (this is the slow part)
echo "⏳ Installing mmengine and mmcv (this may take 5-10 minutes)..."
mim install "mmengine>=0.9.0"

# For CUDA 12.1, use pre-built wheels if available, otherwise build
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Installing mmcv with CUDA support..."
    mim install "mmcv>=2.0.0,<2.2.0"
else
    echo "Installing mmcv for CPU..."
    pip install "mmcv>=2.0.0,<2.2.0"
fi

# Install mmpose
echo "📦 Installing mmpose..."
mim install "mmpose>=1.3.0"

# Install mmdet (required by some mmpose models)
echo "📦 Installing mmdet..."
mim install "mmdet>=3.0.0,<3.3.0"

# Verify installation
echo "✅ Verifying installation..."
python -c "
import mmpose
import mmcv
import mmengine
import mmdet
print(f'✅ mmpose: {mmpose.__version__}')
print(f'✅ mmcv: {mmcv.__version__}')
print(f'✅ mmengine: {mmengine.__version__}')
print(f'✅ mmdet: {mmdet.__version__}')
"

echo ""
echo "🎉 MMPose installation complete!"
echo ""
echo "Next steps:"
echo "1. Convert dataset: python tools/convert_dogflw_to_coco.py ..."
echo "2. Train: python tools/train.py configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py ..."

