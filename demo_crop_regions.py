"""Demo script to show cropping of individual facial regions."""
import os
import cv2
import numpy as np
import json
from config import TRAIN_IMAGES, TRAIN_ANN
from crop_facial_regions import DogFacialRegionCropper


def demo_region_cropping():
    """Demonstrate cropping individual facial regions."""
    # Load annotations
    with open(TRAIN_ANN, 'r') as f:
        coco_data = json.load(f)
    
    # Get first annotation
    ann = coco_data['annotations'][0]
    img_info = next(img for img in coco_data['images'] if img['id'] == ann['image_id'])
    
    # Load image
    img_path = os.path.join(TRAIN_IMAGES, img_info['file_name'])
    img = cv2.imread(img_path)
    
    # Convert keypoints
    keypoints_flat = ann['keypoints']
    keypoints = []
    for i in range(0, len(keypoints_flat), 3):
        x, y, v = keypoints_flat[i:i+3]
        conf = 1.0 if v > 0 else 0.0
        keypoints.append([x, y, conf])
    keypoints = np.array(keypoints)
    
    # Initialize cropper
    cropper = DogFacialRegionCropper()
    
    # Create output directory
    output_dir = "cropped_regions_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original with bounding boxes
    cropper.visualize_regions(img, keypoints, 
                             os.path.join(output_dir, "original_with_boxes.jpg"),
                             padding=0.1)
    
    # Crop all regions
    crops = cropper.crop_all_regions(img, keypoints, padding=0.1)
    
    # Save individual crops
    print(f"\nCropping regions from: {img_info['file_name']}")
    print("-" * 50)
    
    for region_name, crop_img in crops.items():
        output_path = os.path.join(output_dir, f"{region_name}.jpg")
        cv2.imwrite(output_path, crop_img)
        h, w = crop_img.shape[:2]
        print(f"✓ {region_name:12s} - Size: {w}x{h} pixels")
    
    # Create a grid view of all crops
    create_crop_grid(crops, output_dir)
    
    print("\n✅ Done! Check the 'cropped_regions_demo' folder for:")
    print("   - original_with_boxes.jpg: Original image with region bounding boxes")
    print("   - [region_name].jpg: Individual cropped regions")
    print("   - crops_grid.jpg: All crops in a grid layout")


def create_crop_grid(crops, output_dir):
    """Create a grid showing all cropped regions."""
    # Target size for each crop in grid
    target_size = (200, 200)
    
    # Layout: 3x3 grid
    # [left_ear] [left_eye ] [right_eye] [right_ear]
    # [        ] [forehead ] [forehead ] [         ]
    # [        ] [  nose   ] [  nose   ] [         ]
    # [        ] [  mouth  ] [  mouth  ] [         ]
    
    grid = np.ones((target_size[1] * 4, target_size[0] * 4, 3), dtype=np.uint8) * 240
    
    # Place crops in grid
    positions = {
        'left_ear': (0, 0),
        'left_eye': (0, 1),
        'right_eye': (0, 2),
        'right_ear': (0, 3),
        'forehead': (1, 1, 2),  # span 2 columns
        'nose': (2, 1, 2),      # span 2 columns
        'mouth': (3, 1, 2)      # span 2 columns
    }
    
    for region, pos in positions.items():
        if region in crops:
            row = pos[0]
            col = pos[1]
            span = pos[2] if len(pos) > 2 else 1
            
            # Resize crop
            crop = crops[region]
            if span > 1:
                # For spanning regions, make them wider
                resized = cv2.resize(crop, (target_size[0] * span, target_size[1]))
            else:
                resized = cv2.resize(crop, target_size)
            
            # Place in grid
            y1 = row * target_size[1]
            y2 = y1 + target_size[1]
            x1 = col * target_size[0]
            x2 = x1 + target_size[0] * span
            
            grid[y1:y2, x1:x2] = resized
            
            # Add label
            label = region.replace('_', ' ').title()
            cv2.putText(grid, label, (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(os.path.join(output_dir, "crops_grid.jpg"), grid)


if __name__ == "__main__":
    demo_region_cropping()
