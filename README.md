# DogFLW Face Landmark Detection

Clean PyTorch implementation for training dog facial landmark detection on the DogFLW dataset (46 keypoints).

## Features

- ✅ **Simple PyTorch trainer** - No MMPose dependency hell
- ✅ **HRNet-W32 backbone** - Using timm for easy model loading
- ✅ **COCO format annotations** - Standard format, easy to work with
- ✅ **Heatmap-based pose estimation** - Gaussian heatmaps for landmark prediction
- ✅ **Clean, readable code** - ~500 lines total

## Dataset

**DogFLW**: 4,335 images with 46 facial landmarks per dog face
- **Train**: 3,853 images
- **Val**: 479 images  
- **License**: CC BY-NC 4.0 (non-commercial)

## Setup

### 1. Install Dependencies

**Requirements:** Python 3.11 or 3.12

#### Option A: Local Development

```bash
# Initialize project with uv
uv init --python 3.11  # or 3.12
uv sync

# Add required packages
uv add "torch>=2.1.0,<2.3.0" "torchvision>=0.16.0,<0.18.0" "numpy<2.0"
uv add pillow opencv-python-headless tqdm timm kagglehub scikit-image
```

#### Option B: SageMaker / Cloud (Recommended)

```bash
# Clone the repo
git clone https://github.com/naruto716/dogface_landmark_estimation_hrcnn.git
cd dogface_landmark_estimation_hrcnn

# Install from requirements.txt (flexible version constraints for Python 3.11/3.12)
uv pip install -r requirements.txt

# Or with standard pip (no uv needed)
pip install -r requirements.txt
```

**Important for SageMaker/Headless Servers:**
- Use `opencv-python-headless` (no GUI dependencies)  
- Use `numpy<2.0` (compatible with PyTorch 2.x)
- Works with Python 3.11 or 3.12

### 2. Download Dataset

```bash
uv run python -c "
import kagglehub
path = kagglehub.dataset_download('georgemartvel/dogflw')
print('Dataset downloaded to:', path)
"
```

### 3. Convert to COCO Format

```bash
uv run python tools/convert_dogflw_to_coco.py \
  --dogflw_root ~/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW \
  --out_dir data/dogflw/annotations
```

## Training

### Quick Start

```bash
uv run python train_simple.py \
  --epochs 210 \
  --batch-size 16 \
  --lr 1e-3 \
  --work-dir work_dirs/dogflw_hrnet
```

### Training Arguments

- `--data-root`: Path to DogFLW dataset (default: auto-detected from kagglehub)
- `--ann-root`: Path to COCO annotations (default: `data/dogflw/annotations`)
- `--epochs`: Number of training epochs (default: 210)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--num-workers`: DataLoader workers (default: 4)
- `--device`: Device to use (default: auto-detect cuda/mps/cpu)

### What Gets Trained

- **Model**: HRNet-W32 + simple conv head (46 keypoint heatmaps)
- **Input**: 256×256 RGB images
- **Output**: 128×128 heatmaps (one per keypoint)
- **Loss**: MSE loss with visibility weighting
- **Optimizer**: Adam with cosine annealing
- **Metric**: Normalized Mean Error (NME) normalized by image size

## Project Structure

```
760face/
├── dataset.py                 # DogFLW dataset loader (COCO format)
├── train_simple.py            # Training script (~300 lines)
├── tools/
│   └── convert_dogflw_to_coco.py  # Dataset converter
├── data/
│   └── dogflw/
│       └── annotations/       # COCO JSON files (train.json, val.json)
└── work_dirs/                 # Training outputs (checkpoints, logs)
```

## Model Architecture

```python
HRNet-W32 (pretrained on ImageNet)
    ↓
[B, 64, H/2, W/2] features
    ↓
Conv3x3 + BN + ReLU (64→64)
    ↓
Conv1x1 (64→46)
    ↓
[B, 46, H/2, W/2] heatmaps
```

## Results

Training for 210 epochs should give:
- **NME**: ~0.03-0.05 (3-5% of image size)
- **Time**: ~2-4 hours on modern GPU

Checkpoints saved:
- `work_dirs/dogflw_hrnet/best_model.pth` - Best validation NME
- `work_dirs/dogflw_hrnet/epoch_*.pth` - Every 10 epochs

## Notes

### Why Not MMPose?

We initially tried MMPose but ran into:
- ❌ **Dependency hell** - MMCV requires exact PyTorch + Python + CUDA versions
- ❌ **Long build times** - MMCV builds from source take 20-30 minutes
- ❌ **ABI incompatibilities** - Pre-built wheels don't always work
- ❌ **Overcomplicated** - Too much abstraction for a simple task

Our custom PyTorch implementation:
- ✅ **Just works** - Standard PyTorch + timm
- ✅ **Fast setup** - Install in < 1 minute
- ✅ **Easy to debug** - Clean, readable code
- ✅ **Easy to extend** - Modify training loop, augmentations, etc.

### Checkpoints

**Large files (>100MB) are not included in this repo** due to GitHub limits.

The HRNet-W32 pretrained weights will be automatically downloaded by `timm` when you run training.

Your trained checkpoints will be saved to `work_dirs/` (which is gitignored).

## Troubleshooting

### SageMaker / Cloud Issues

**Error: `No solution found when resolving dependencies` (Python 3.12)**
```bash
# SageMaker often uses Python 3.12, which needs PyTorch 2.2+
# The flexible requirements.txt handles this automatically
git pull  # Get latest requirements.txt
pip install -r requirements.txt
```

**Error: `ImportError: libGL.so.1: cannot open shared object file`**
```bash
# Solution: Use opencv-python-headless instead
pip uninstall opencv-python
pip install opencv-python-headless
```

**Error: `NumPy 1.x cannot be run in NumPy 2.x`**
```bash
# Solution: Use numpy<2.0 (already in requirements.txt)
pip install "numpy>=1.24.0,<2.0"
```

**One-liner fix for all common issues:**
```bash
git pull && pip install -r requirements.txt --force-reinstall --no-cache-dir
```

### Memory Issues

If you run out of memory:
```bash
# Reduce batch size
uv run python train_simple.py --batch-size 8  # or even 4

# Reduce num_workers
uv run python train_simple.py --num-workers 0
```

## License

- **Code**: MIT (this implementation)
- **DogFLW Dataset**: CC BY-NC 4.0 (non-commercial)

For commercial use, you'll need to obtain permission from the DogFLW authors or create your own dataset.

## References

- [DogFLW Dataset](https://github.com/martvelge/DogFLW)
- [HRNet Paper](https://arxiv.org/abs/1902.09212)
- [timm Library](https://github.com/huggingface/pytorch-image-models)

## Citation

If you use DogFLW, please cite:

```bibtex
@misc{dogflw2024,
  author = {Martvelge},
  title = {Dog Facial Landmarks in the Wild},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/martvelge/DogFLW}
}
```
