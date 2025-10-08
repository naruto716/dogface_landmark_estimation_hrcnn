# Integration Guide

How to integrate this package into your project.

## Option 1: As a Submodule (Recommended)

```bash
# In your project directory
git submodule add <your-repo-url>/dog_face_inference dog_face_inference
cd dog_face_inference
bash setup.sh
```

Then in your code:
```python
from dog_face_inference import DogFacePredictor

predictor = DogFacePredictor(
    config_path='dog_face_inference/configs/dog_face_config.py',
    checkpoint_path='dog_face_inference/models/dog_face_model.pth'
)
```

## Option 2: Copy to Your Project

```bash
# Copy the entire folder
cp -r dog_face_inference /path/to/your/project/

# Install dependencies
cd /path/to/your/project/dog_face_inference
pip install -r requirements.txt
```

## Option 3: As a Python Package

Make it installable:

```bash
# In dog_face_inference directory, create setup.py:
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name='dog-face-inference',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy<2.0',
        'opencv-python-headless>=4.8.0',
        'mmengine>=0.8.0',
        'mmcv>=2.0.0,<2.2.0',
        'mmpose>=1.0.0',
        'mmdet>=3.0.0',
    ],
)
EOF

# Install in your project's environment
pip install -e .
```

Then use anywhere:
```python
from dog_face_inference import DogFacePredictor
```

## Minimal Integration Example

```python
# your_project/main.py

import sys
sys.path.append('dog_face_inference')  # If not installed

from dog_face_predictor import DogFacePredictor

# Initialize once (at startup)
dog_detector = DogFacePredictor(
    config_path='dog_face_inference/configs/dog_face_config.py',
    checkpoint_path='dog_face_inference/models/dog_face_model.pth',
    device='cuda:0'
)

def process_dog_image(image_path):
    """Your main processing function"""
    # Detect landmarks and extract regions
    result = dog_detector.predict(image_path)
    
    # Use the results
    if result['avg_confidence'] > 0.7:
        # High confidence - process the regions
        for region_name, region_img in result['regions'].items():
            # Your custom logic here
            analyze_region(region_name, region_img)
        
        return result
    else:
        # Low confidence - handle accordingly
        return None

def analyze_region(region_name, region_img):
    """Your custom region processing"""
    # Example: Feed to another model, save to database, etc.
    pass
```

## API Integration

If you're building an API:

```python
# api.py
from flask import Flask, request, jsonify
from dog_face_inference import DogFacePredictor
import base64
import cv2
import numpy as np

app = Flask(__name__)

# Initialize predictor once at startup
predictor = DogFacePredictor(
    config_path='configs/dog_face_config.py',
    checkpoint_path='models/dog_face_model.pth',
    device='cuda:0'
)

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    image_file = request.files['image']
    image_path = save_uploaded_file(image_file)
    
    # Run inference
    result = predictor.predict(image_path, extract_regions=False)
    
    # Return JSON response
    return jsonify({
        'landmarks': result['landmarks'].tolist(),
        'confidence': result['avg_confidence'],
        'visible_landmarks': result['visible_landmarks']
    })

@app.route('/extract_regions', methods=['POST'])
def extract_regions():
    image_file = request.files['image']
    image_path = save_uploaded_file(image_file)
    
    result = predictor.predict(image_path, extract_regions=True)
    
    # Encode regions as base64 for JSON response
    encoded_regions = {}
    for name, img in result['regions'].items():
        _, buffer = cv2.imencode('.jpg', img)
        encoded_regions[name] = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'confidence': result['avg_confidence'],
        'regions': encoded_regions
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Docker Integration

Create a Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy inference package
COPY dog_face_inference /app/dog_face_inference

# Install dependencies
RUN pip install -r dog_face_inference/requirements.txt

# Copy your application
COPY your_app.py /app/

# Run
CMD ["python", "your_app.py"]
```

## Performance Tips

### 1. Initialize Once
```python
# ✅ Good - Initialize at startup
predictor = DogFacePredictor(...)

def process_image(path):
    return predictor.predict(path)
```

```python
# ❌ Bad - Initialize every time
def process_image(path):
    predictor = DogFacePredictor(...)  # Slow!
    return predictor.predict(path)
```

### 2. Batch Processing
```python
# Process multiple images efficiently
results = predictor.predict_batch(image_paths)
```

### 3. Skip Regions When Not Needed
```python
# Faster if you only need landmarks
result = predictor.predict(path, extract_regions=False)
```

## Common Issues

### Issue: Import errors
```python
# Solution: Add to Python path
import sys
sys.path.append('path/to/dog_face_inference')
```

### Issue: CUDA out of memory
```python
# Solution: Use CPU or smaller batch size
predictor = DogFacePredictor(..., device='cpu')
```

### Issue: Model checkpoint mismatch
```bash
# Solution: Ensure you're using the correct checkpoint
# Must be trained on DogFLW (46 keypoints), not COCO (17 keypoints)
```

## Questions?

See README.md for full documentation and examples.
