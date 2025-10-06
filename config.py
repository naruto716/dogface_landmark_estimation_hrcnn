"""Configuration for local development - easier than typing long paths!"""
import os

# Path to your local DogFLW dataset
# Change this if you move the data elsewhere
DOGFLW_ROOT = os.path.expanduser('~/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW')

# Or if you want to copy/move the data locally:
# DOGFLW_ROOT = './data/DogFLW'

# Annotation paths
ANN_ROOT = 'data/dogflw/annotations'

# Quick access paths
TRAIN_IMAGES = os.path.join(DOGFLW_ROOT, 'train/images')
TEST_IMAGES = os.path.join(DOGFLW_ROOT, 'test/images')
TRAIN_LABELS = os.path.join(DOGFLW_ROOT, 'train/labels')
TEST_LABELS = os.path.join(DOGFLW_ROOT, 'test/labels')
TRAIN_ANN = os.path.join(ANN_ROOT, 'train.json')
VAL_ANN = os.path.join(ANN_ROOT, 'val.json')

# Print info when imported
if __name__ == '__main__':
    print(f"DogFLW root: {DOGFLW_ROOT}")
    print(f"Train images: {TRAIN_IMAGES}")
    print(f"Test images: {TEST_IMAGES}")
    print(f"Annotations: {ANN_ROOT}")
    
    # Check if paths exist
    import os
    if os.path.exists(DOGFLW_ROOT):
        print("✅ DogFLW data found!")
    else:
        print("❌ DogFLW data not found at configured path")
