"""Run trained MMPose model on new dog images."""
import os
import sys
import random
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample, merge_data_samples


def get_random_images(root_dir, num_images=100):
    """Get first N images from PetFace dog dataset (stop early, don't scan all 190k!)."""
    images = []
    root_path = Path(root_dir)
    
    # Stop as soon as we have enough images
    for dog_folder in sorted(root_path.iterdir()):
        if not dog_folder.is_dir():
            continue
            
        for img_file in sorted(dog_folder.iterdir()):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                images.append(str(img_file))
                
                if len(images) >= num_images:
                    print(f"Collected {len(images)} images")
                    return images
    
    print(f"Collected {len(images)} images")
    return images


def detect_dog_bbox(image_path):
    """Simple bbox estimation - use whole image or center crop."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Use whole image as bbox with some padding
    bbox = np.array([0, 0, w, h])
    return bbox, img


def visualize_landmarks(img, keypoints, output_path):
    """Draw landmarks on image and save."""
    vis_img = img.copy()
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # Only draw confident predictions
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.circle(vis_img, (int(x), int(y)), 3, color, -1)
            # Optionally add numbers
            # cv2.putText(vis_img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    
    cv2.imwrite(output_path, vis_img)
    return vis_img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--img-dir', required=True, help='Directory containing dog images')
    parser.add_argument('--output-dir', default='inference_results', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of images to process')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    
    # Get random images
    print(f"Sampling {args.num_samples} images from {args.img_dir}")
    image_paths = get_random_images(args.img_dir, args.num_samples)
    
    # Process each image
    results = []
    print(f"\nProcessing {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths):
        try:
            # Get bbox (simple version - use whole image)
            bbox, img = detect_dog_bbox(img_path)
            h, w = img.shape[:2]
            
            # Create bbox in COCO format [x, y, w, h]
            bbox_coco = np.array([[0, 0, w, h, 1.0]])  # [x, y, w, h, score]
            
            # Run inference
            pose_results = inference_topdown(model, img_path, bboxes=bbox_coco)
            
            if len(pose_results) > 0:
                # Get keypoints from first detection
                keypoints = pose_results[0].pred_instances.keypoints[0]  # [46, 2]
                scores = pose_results[0].pred_instances.keypoint_scores[0]  # [46]
                
                # Combine into [46, 3] format
                keypoints_with_conf = np.concatenate([
                    keypoints, scores.reshape(-1, 1)
                ], axis=1)
                
                # Save visualization
                img_name = Path(img_path).stem
                dog_id = Path(img_path).parent.name
                output_path = os.path.join(args.output_dir, f'{dog_id}_{img_name}.jpg')
                visualize_landmarks(img, keypoints_with_conf, output_path)
                
                # Save prediction data
                results.append({
                    'image_path': img_path,
                    'dog_id': dog_id,
                    'keypoints': keypoints.tolist(),
                    'scores': scores.tolist(),
                    'avg_confidence': float(scores.mean())
                })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Save all predictions to JSON
    output_json = os.path.join(args.output_dir, 'predictions.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Inference Complete!")
    print(f"{'='*50}")
    print(f"Processed: {len(results)} / {len(image_paths)} images")
    print(f"Visualizations saved to: {args.output_dir}")
    print(f"Predictions saved to: {output_json}")
    
    if results:
        avg_conf = np.mean([r['avg_confidence'] for r in results])
        print(f"Average confidence: {avg_conf:.3f}")
    
    # Show some sample predictions
    print(f"\nSample predictions (first 3):")
    for r in results[:3]:
        print(f"  {Path(r['image_path']).name}: conf={r['avg_confidence']:.3f}")


if __name__ == '__main__':
    main()

