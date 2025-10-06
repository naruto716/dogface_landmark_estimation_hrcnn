"""Visualize just the numbered landmarks without bounding boxes for manual mapping."""
import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
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


def visualize_landmarks_only(data_root, ann_file, output_dir, num_samples=10):
    """Process images and show only numbered landmarks."""
    
    # Load annotations
    print(f"Loading annotations from {ann_file}")
    coco_data, img_to_ann, id_to_img = load_coco_annotations(ann_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
            # Convert visibility
            conf = 1.0 if v > 0 else 0.0
            keypoints.append([x, y, conf])
        
        keypoints = np.array(keypoints)
        
        # Create visualization
        vis_img = img.copy()
        
        # Draw all landmarks with numbers
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf > 0:
                # Draw a small filled circle
                cv2.circle(vis_img, (int(x), int(y)), 4, (0, 255, 0), -1)
                # Draw white outline for contrast
                cv2.circle(vis_img, (int(x), int(y)), 4, (255, 255, 255), 1)
                
                # Add number with background for better visibility
                text = str(idx)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Position text slightly offset from the point
                text_x = int(x) + 7
                text_y = int(y) - 7
                
                # Draw white background rectangle for the text
                cv2.rectangle(vis_img, 
                            (text_x - 2, text_y - text_height - 2),
                            (text_x + text_width + 2, text_y + 2),
                            (255, 255, 255), -1)
                
                # Draw the number in black
                cv2.putText(vis_img, text, (text_x, text_y),
                           font, font_scale, (0, 0, 0), thickness)
        
        # Save visualization
        output_path = os.path.join(output_dir, f'landmarks_{img_info["file_name"]}')
        cv2.imwrite(output_path, vis_img)
        
        # Also save original for reference
        original_path = os.path.join(output_dir, f'original_{img_info["file_name"]}')
        cv2.imwrite(original_path, img)
        
        processed += 1
        print(f"Processed {processed}/{num_samples}: {img_info['file_name']}")
    
    print(f"\n✅ Processed {processed} images")
    print(f"📁 Results saved to: {output_dir}")
    
    # Create a summary image showing all landmarks in a grid
    create_landmark_grid(output_dir, num_keypoints)


def create_landmark_grid(output_dir, num_keypoints=46):
    """Create a grid showing all landmark numbers for reference."""
    # Create a large canvas
    grid_size = 1200
    canvas = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Title
    cv2.putText(canvas, "Dog Face Landmarks Reference (0-45)", (300, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Draw a rough dog face outline for context
    center_x, center_y = grid_size // 2, grid_size // 2 + 50
    
    # Face outline (ellipse)
    cv2.ellipse(canvas, (center_x, center_y), (300, 250), 0, 0, 360, (200, 200, 200), 2)
    
    # Add text regions
    regions = [
        ("Left Eye Area: ?", (100, 150)),
        ("Right Eye Area: ?", (800, 150)),
        ("Nose Area: ?", (center_x - 50, 300)),
        ("Mouth Area: ?", (center_x - 50, 600)),
        ("Left Ear Area: ?", (100, 400)),
        ("Right Ear Area: ?", (800, 400)),
        ("Face Contour: ?", (center_x - 50, 850))
    ]
    
    for text, (x, y) in regions:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
    
    # Save the reference
    ref_path = os.path.join(output_dir, 'landmark_reference_grid.png')
    cv2.imwrite(ref_path, canvas)
    print(f"Created landmark reference: {ref_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',
                       default='landmarks_only',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples',
                       type=int,
                       default=10,
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
    visualize_landmarks_only(img_dir, ann_file, args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()
