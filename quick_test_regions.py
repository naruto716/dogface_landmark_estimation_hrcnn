#!/usr/bin/env python3
"""Quick test of region groupings - no arguments needed!"""
import os
import sys

# Import config first to check if data exists
try:
    from config import DOGFLW_ROOT, TRAIN_IMAGES, TRAIN_ANN
    
    # Quick check if data exists
    if not os.path.exists(DOGFLW_ROOT):
        print("❌ DogFLW data not found!")
        print(f"   Expected at: {DOGFLW_ROOT}")
        print("\nTo download the data:")
        print("  python -c 'import kagglehub; kagglehub.dataset_download(\"georgemartvel/dogflw\")'")
        sys.exit(1)
        
except ImportError:
    print("❌ config.py not found!")
    sys.exit(1)

# Now run the test
print("🐕 Testing facial region groupings...")
print(f"Using data from: {DOGFLW_ROOT}")
print("")

# Import and run the test
from test_region_groupings import process_images, create_landmark_reference_image

# Process 10 images by default
output_dir = "quick_test_output"
num_samples = 10

print(f"Processing {num_samples} images...")
process_images(TRAIN_IMAGES, TRAIN_ANN, output_dir, num_samples)
create_landmark_reference_image(output_dir)

print("\n✅ Done! Check the 'quick_test_output' folder for results:")
print("  - comparison_*.png: Side-by-side original vs regions")  
print("  - regions_*.png: Images with bounding boxes and landmark indices")
print("  - landmark_index_reference.png: Reference showing all landmark numbers")
print("  - current_region_mapping.json: Current landmark groupings")
print("\nOpen these images to see if the landmark groupings look correct!")
