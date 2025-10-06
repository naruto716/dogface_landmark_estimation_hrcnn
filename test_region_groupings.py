"""Test facial region groupings by visualizing bounding boxes on ground truth data."""
import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from crop_facial_regions import DogFacialRegionCropper
from config import DOGFLW_ROOT, TRAIN_IMAGES, TEST_IMAGES, TRAIN_ANN, VAL_ANN


def load_coco_annotations(ann_file):
    """Load COCO format annotations."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Create image id to annotation mapping
    img_to_ann = {}
    for ann in data['annotations']:
        img_to_ann[ann['image_id']] = ann
    
    # Create image id to filename mapping
    id_to_img = {img['id']: img for img in data['images']}
    
    return data, img_to_ann, id_to_img


def process_images(data_root, ann_file, output_dir, num_samples=20):
    """Process images and visualize region groupings."""
    
    # Load annotations
    print(f"Loading annotations from {ann_file}")
    coco_data, img_to_ann, id_to_img = load_coco_annotations(ann_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize cropper with default regions
    cropper = DogFacialRegionCropper()
    
    # Save the current region mapping
    region_map_file = os.path.join(output_dir, 'current_region_mapping.json')
    with open(region_map_file, 'w') as f:
        json.dump(cropper.landmark_regions, f, indent=2)
    print(f"Saved current region mapping to {region_map_file}")
    
    # Process sample images
    processed = 0
    
    for img_id, img_info in id_to_img.items():
        if processed >= num_samples:
            break
            
        if img_id not in img_to_ann:
            continue
            
        # Load image
        img_path = os.path.join(data_root, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get annotations
        ann = img_to_ann[img_id]
        
        # Convert keypoints to our format
        keypoints_flat = ann['keypoints']
        num_keypoints = len(keypoints_flat) // 3
        
        keypoints = []
        for i in range(num_keypoints):
            x = keypoints_flat[i * 3]
            y = keypoints_flat[i * 3 + 1]
            v = keypoints_flat[i * 3 + 2]
            # Convert visibility: COCO uses 0=not labeled, 1=labeled but not visible, 2=labeled and visible
            # We'll use confidence: 0 for not visible, 1 for visible
            conf = 1.0 if v > 0 else 0.0
            keypoints.append([x, y, conf])
        
        keypoints = np.array(keypoints)
        
        # Create visualization
        vis_img = img.copy()
        
        # Define colors for each region
        region_colors = {
            'left_eye': (255, 0, 0),      # Blue in BGR
            'right_eye': (255, 255, 0),    # Cyan
            'nose': (0, 0, 255),           # Red
            'mouth': (0, 165, 255),        # Orange
            'left_ear': (0, 255, 0),       # Green
            'right_ear': (0, 255, 127),    # Light green
            'forehead': (128, 0, 128)      # Purple
        }
        
        # Draw bounding boxes and landmarks for each region
        for region, color in region_colors.items():
            # Get bbox for this region
            bbox = cropper.get_region_bbox(keypoints, region, padding=0.1)
            
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color, 1)  # Thin boundary
                
                # Add label
                label = region.replace('_', ' ').title()
                cv2.putText(vis_img, label, (x_min, y_min - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw landmarks for this region
            if region in cropper.landmark_regions:
                for idx in cropper.landmark_regions[region]:
                    if idx < len(keypoints):
                        x, y, conf = keypoints[idx]
                        if conf > 0:
                            # Draw landmark with region color
                            cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)
                            # Add landmark index
                            cv2.putText(vis_img, str(idx), (int(x) + 5, int(y) - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save visualization
        output_path = os.path.join(output_dir, f'regions_{img_info["file_name"]}')
        cv2.imwrite(output_path, vis_img)
        
        # Also create a comparison image showing original vs regions
        comparison = np.hstack([img, vis_img])
        comp_path = os.path.join(output_dir, f'comparison_{img_info["file_name"]}')
        cv2.imwrite(comp_path, comparison)
        
        processed += 1
        print(f"Processed {processed}/{num_samples}: {img_info['file_name']}")
    
    print(f"\n✅ Processed {processed} images")
    print(f"📁 Results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the visualizations to check if landmark groupings are correct")
    print("2. Update the region mapping in crop_facial_regions.py if needed")
    print("3. Run this script again to verify corrections")


def create_landmark_reference_image(output_dir):
    """Create a reference image showing all landmark indices."""
    # Create a blank image
    img_size = 800
    ref_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Create a grid of landmark positions (7x7 grid for 46 landmarks)
    grid_size = 7
    margin = 100
    spacing = (img_size - 2 * margin) // (grid_size - 1)
    
    landmark_positions = []
    idx = 0
    
    for row in range(grid_size):
        for col in range(grid_size):
            if idx >= 46:
                break
            x = margin + col * spacing
            y = margin + row * spacing
            
            # Draw circle
            cv2.circle(ref_img, (x, y), 20, (0, 0, 0), -1)
            
            # Add index
            cv2.putText(ref_img, str(idx), (x - 10, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            idx += 1
    
    # Add title
    cv2.putText(ref_img, "Landmark Index Reference (0-45)", (250, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    ref_path = os.path.join(output_dir, 'landmark_index_reference.png')
    cv2.imwrite(ref_path, ref_img)
    print(f"Created landmark reference image: {ref_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',
                       default='region_grouping_test',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples',
                       type=int,
                       default=20,
                       help='Number of images to process')
    parser.add_argument('--split',
                       default='train',
                       choices=['train', 'test'],
                       help='Which split to use')
    args = parser.parse_args()
    
    # Use paths from config.py
    if args.split == 'train':
        img_dir = TRAIN_IMAGES
        ann_file = TRAIN_ANN
    else:
        img_dir = TEST_IMAGES
        ann_file = VAL_ANN
    
    # Process images
    process_images(img_dir, ann_file, args.output_dir, args.num_samples)
    
    # Create reference image
    create_landmark_reference_image(args.output_dir)


if __name__ == '__main__':
    main()
