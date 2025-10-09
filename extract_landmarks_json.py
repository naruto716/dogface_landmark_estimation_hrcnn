"""
Extract landmarks and region bboxes to JSON for training pipeline.
Output: One JSON file per image with landmarks and bbox coordinates.
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from mmpose.apis import init_model, inference_topdown
from crop_facial_regions import DogFacialRegionCropper


def process_image(model, cropper, image_path):
    """Process single image and return landmarks + bboxes as dict."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    bbox_xyxy = np.array([[0, 0, w, h]])
    
    # Run inference
    pose_results = inference_topdown(model, image_path, bboxes=bbox_xyxy)
    
    if len(pose_results) == 0:
        return None
    
    # Extract keypoints
    pred_instances = pose_results[0].pred_instances
    keypoints = pred_instances.keypoints
    scores = pred_instances.keypoint_scores
    
    # Convert to numpy
    if hasattr(keypoints, 'cpu'):
        keypoints = keypoints.cpu().numpy()
    if hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
    
    # Handle dimensions
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]
    
    # Combine into (46, 3) format
    keypoints_with_conf = np.concatenate([keypoints, scores.reshape(-1, 1)], axis=1)
    
    # Get region bounding boxes
    region_bboxes = {}
    for region_name in cropper.landmark_regions.keys():
        bbox = cropper.get_region_bbox(keypoints_with_conf, region_name, padding=0.1)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            region_bboxes[region_name] = {
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max),
                'width': int(x_max - x_min),
                'height': int(y_max - y_min)
            }
    
    # Prepare output
    result = {
        'image_path': str(image_path),
        'image_width': w,
        'image_height': h,
        'landmarks': [
            {
                'id': i,
                'x': float(keypoints_with_conf[i, 0]),
                'y': float(keypoints_with_conf[i, 1]),
                'confidence': float(keypoints_with_conf[i, 2])
            }
            for i in range(len(keypoints_with_conf))
        ],
        'region_bboxes': region_bboxes,
        'avg_confidence': float(scores.mean()),
        'visible_landmarks': int((scores > 0.3).sum())
    }
    
    return result


def get_images(root_dir, max_images=None):
    """Get all images from directory (handles both flat and nested structure)."""
    images = []
    root_path = Path(root_dir)
    
    # Try direct files first
    for img_file in root_path.glob('*'):
        if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            images.append(str(img_file))
            if max_images and len(images) >= max_images:
                return images
    
    # If no images found, look in subfolders
    if not images:
        for img_file in root_path.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                images.append(str(img_file))
                if max_images and len(images) >= max_images:
                    return images
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description='Extract landmarks and region bboxes to JSON files'
    )
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--img-dir', required=True, help='Directory containing images')
    parser.add_argument('--output-dir', required=True, help='Output directory for JSON files')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--device', default='cuda:0', help='Device (cuda:0 or cpu)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    # Initialize cropper
    cropper = DogFacialRegionCropper()
    
    # Get images
    print(f"Scanning {args.img_dir} for images...")
    image_paths = get_images(args.img_dir, args.max_images)
    print(f"Found {len(image_paths)} images")
    
    # Process images
    print(f"\nProcessing images...")
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_paths):
        try:
            # Process image
            result = process_image(model, cropper, img_path)
            
            if result is None:
                failed += 1
                continue
            
            # Save JSON with dog ID to avoid overwriting
            img_path_obj = Path(img_path)
            dog_id = img_path_obj.parent.name  # Get parent folder name (dog ID)
            img_name = img_path_obj.stem
            output_file = os.path.join(args.output_dir, f'{dog_id}_{img_name}.json')
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            failed += 1
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(image_paths)}")
    print(f"Failed: {failed}/{len(image_paths)}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nEach JSON contains:")
    print(f"  - landmarks: 46 keypoints with x, y, confidence")
    print(f"  - region_bboxes: Bounding boxes for 7 facial regions")
    print(f"  - image dimensions and metadata")
    
    # Create summary JSON
    summary = {
        'total_images': len(image_paths),
        'successful': successful,
        'failed': failed,
        'output_directory': args.output_dir,
        'regions': list(cropper.landmark_regions.keys())
    }
    
    summary_file = os.path.join(args.output_dir, '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()

