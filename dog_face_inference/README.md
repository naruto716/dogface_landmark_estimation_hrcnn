# Dog Face Landmark Inference Package

Clean, minimal package for running inference on dog faces to detect 46 facial landmarks and extract facial regions.

## Features

- 🎯 Detect 46 facial landmarks on dog faces
- ✂️ Extract 7 facial regions: eyes, nose, mouth, ears, forehead
- 📦 Simple API - just a few lines of code
- 🚀 Fast inference with GPU support

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch torchvision mmengine mmcv mmpose mmdet opencv-python-headless numpy

# Or use the requirements file
pip install -r requirements.txt
```

### 2. Download Model

Place your trained model checkpoint in `models/`:
```
dog_face_inference/
├── models/
│   └── dog_face_model.pth
```

### 3. Run Inference

```python
from dog_face_predictor import DogFacePredictor

# Initialize predictor
predictor = DogFacePredictor(
    config_path='configs/dog_face_config.py',
    checkpoint_path='models/dog_face_model.pth',
    device='cuda:0'  # or 'cpu'
)

# Predict on single image
result = predictor.predict('path/to/dog_image.jpg')

# Access results
landmarks = result['landmarks']          # (46, 3) array with x, y, confidence
regions = result['regions']              # Dict of cropped regions
confidence = result['avg_confidence']    # Average landmark confidence

# Save visualizations
predictor.save_visualization(
    'path/to/dog_image.jpg',
    'output_folder/'
)
```

## Output Structure

For each image, you get:

```python
{
    'landmarks': np.array,           # (46, 3) - x, y, confidence for each landmark
    'regions': {
        'left_eye': np.array,        # Cropped left eye image
        'right_eye': np.array,       # Cropped right eye image
        'nose': np.array,            # Cropped nose image
        'mouth': np.array,           # Cropped mouth image
        'left_ear': np.array,        # Cropped left ear image
        'right_ear': np.array,       # Cropped right ear image
        'forehead': np.array         # Cropped forehead image
    },
    'avg_confidence': float,         # Overall confidence score
    'visible_landmarks': int         # Number of visible landmarks
}
```

## Integration Example

```python
# Batch processing
from dog_face_predictor import DogFacePredictor
import glob

predictor = DogFacePredictor(
    config_path='configs/dog_face_config.py',
    checkpoint_path='models/dog_face_model.pth'
)

# Process multiple images
image_paths = glob.glob('images/*.jpg')
for img_path in image_paths:
    result = predictor.predict(img_path)
    
    # Use the results
    if result['avg_confidence'] > 0.7:
        # High confidence - save regions
        for region_name, region_img in result['regions'].items():
            cv2.imwrite(f'output/{region_name}.jpg', region_img)
```

## Files Needed

Minimal files required for inference:

```
dog_face_inference/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── dog_face_predictor.py       # Main inference class
├── landmark_regions.py          # Facial region definitions
├── configs/
│   └── dog_face_config.py      # Model configuration
└── models/
    └── dog_face_model.pth      # Trained checkpoint (you provide this)
```

## Advanced Usage

### Custom Region Padding

```python
# Adjust padding for cropped regions
result = predictor.predict('image.jpg', region_padding=0.2)  # 20% padding
```

### Get Only Landmarks (No Regions)

```python
result = predictor.predict('image.jpg', extract_regions=False)
landmarks = result['landmarks']
```

### Batch Inference

```python
# Process multiple images efficiently
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'mmengine'`
- Solution: Install MMPose dependencies: `pip install mmengine mmcv mmpose mmdet`

**Issue**: `CUDA out of memory`
- Solution: Use CPU: `predictor = DogFacePredictor(..., device='cpu')`

**Issue**: Model checkpoint mismatch
- Solution: Ensure you're using a checkpoint trained on DogFLW (46 keypoints), not COCO (17 keypoints)

## License

MIT License - Free to use in commercial projects
