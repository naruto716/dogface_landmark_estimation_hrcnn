# DogFLW Face Landmark Detection with MMPose

Minimal, code-first MMPose setup for training dog facial landmark detection on the DogFLW dataset (46 keypoints).

## Dataset

- **DogFLW**: 4,335 images with 46 facial landmarks per dog face
- **Train**: 3,853 images
- **Val**: 479 images  
- **License**: CC BY-NC 4.0 (non-commercial)

## Setup Complete ✅

1. ✅ Dataset downloaded to `~/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW`
2. ✅ Converted to COCO keypoints format → `data/dogflw/annotations/`
3. ✅ MMPose 1.3.2 installed with all dependencies
4. ✅ Config created: `configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py`
5. ✅ Pretrained HRNet-W32 checkpoint downloaded

## Quick Start

### Train

```bash
uv run python tools/train.py \
  configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py \
  --work-dir work_dirs/dogflw_hrnet_w32_udp \
  --load-from checkpoints/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192-73ede547_20220914.pth \
  --amp
```

**Training Details:**
- Model: HRNet-W32 + UDP (Unbiased Data Processing)
- Input: 256×256, Heatmap: 64×64
- Batch size: 16
- Epochs: 210 with cosine annealing
- Optimizer: Adam (lr=3e-4)
- Automatic Mixed Precision enabled

### Evaluate

```bash
uv run python tools/test.py \
  configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py \
  work_dirs/dogflw_hrnet_w32_udp/best_NME_epoch_*.pth \
  --show-dir work_dirs/dogflw_hrnet_w32_udp/vis_val
```

**Evaluation Metric:** NME (Normalized Mean Error) with inter-ocular distance normalization

## Project Structure

```
760face/
├── configs/
│   └── dogflw/
│       └── td-hm_hrnet-w32_udp_dogflw-256x256.py  # Training config
├── data/
│   └── dogflw/
│       └── annotations/
│           ├── train.json  # COCO format
│           └── val.json
├── tools/
│   ├── convert_dogflw_to_coco.py  # Dataset converter
│   ├── train.py                   # Training script
│   └── test.py                    # Evaluation script
├── checkpoints/
│   └── td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192-73ede547_20220914.pth
└── work_dirs/
    └── dogflw_hrnet_w32_udp/      # Training outputs
```

## Model Architecture

- **Backbone**: HRNet-W32 (pretrained on COCO)
- **Head**: Heatmap head with 46 output channels
- **Codec**: UDPHeatmap for unbiased coordinate decoding
- **Loss**: Keypoint MSE Loss with target weighting

## Configuration Highlights

### Data Pipeline
- `LoadImage` → `GetBBoxCenterScale` → `RandomBBoxTransform` (rotation ±30°, scale 0.75-1.25)
- `TopdownAffine` → `GenerateTarget` (UDP heatmaps) → `PackPoseInputs`
- Horizontal flip currently disabled (set `flip_pairs` in config to enable)

### Augmentation
- Random rotation: ±30°
- Random scale: 0.75–1.25×
- Bbox padding: 1.0
- **Note**: Update `flip_pairs` in config with L/R landmark pairs to enable horizontal flipping

## Next Steps (Optional)

1. **Enable horizontal flipping**: Fill `flip_pairs` in config with left/right keypoint indices
2. **Increase input size**: Try `input_size=(320, 320)` if ears are often cropped
3. **Freeze backbone**: Add paramwise LR multipliers for first 5-10 epochs
4. **Create inference script**: Single-image prediction with visualization

## Dependencies

- Python 3.11
- PyTorch 2.8.0
- MMPose 1.3.2
- MMCV 2.2.0
- MMEngine 0.10.7

Managed via `uv` (see `pyproject.toml`)

## References

- [DogFLW Dataset](https://github.com/martvelge/DogFLW)
- [MMPose Documentation](https://mmpose.readthedocs.io/en/latest/)
- [HRNet Paper](https://arxiv.org/abs/1902.09212)
- [UDP Paper](https://arxiv.org/abs/1911.07524)

