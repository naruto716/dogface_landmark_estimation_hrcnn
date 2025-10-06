"""Run trained MMPose model on dog images with facial region extraction."""
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

from crop_facial_regions import DogFacialRegionCropper


def get_random_images(root_dir, num_images=100):
    """Get first N images from PetFace dog dataset."""
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
            # Add landmark numbers
            cv2.putText(vis_img, str(i), (int(x) + 5, int(y) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, vis_img)
    return vis_img


def create_composite_regions(crops, output_path):
    """Create a composite view of all facial regions."""
    # Calculate grid size
    cell_size = 150
    padding = 10
    
    # Create canvas
    canvas_height = cell_size * 3 + padding * 4
    canvas_width = cell_size * 5 + padding * 6
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
    
    # Layout positions
    positions = {
        'left_ear': (0, 0),
        'left_eye': (0, 1),
        'forehead': (0, 2),
        'right_eye': (0, 3),
        'right_ear': (0, 4),
        'nose': (1, 2),
        'mouth': (2, 1, 3)  # row, col, span
    }
    
    # Place each region
    for region, crop in crops.items():
        if region in positions:
            pos = positions[region]
            row = pos[0]
            col = pos[1]
            span = pos[2] if len(pos) > 2 else 1
            
            # Calculate position
            y1 = row * (cell_size + padding) + padding
            x1 = col * (cell_size + padding) + padding
            
            # Resize crop to fit
            if span > 1:
                target_w = cell_size * span + padding * (span - 1)
                target_h = cell_size
            else:
                target_w = cell_size
                target_h = cell_size
            
            # Maintain aspect ratio
            h, w = crop.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(crop, (new_w, new_h))
            
            # Center in cell
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place image - ensure within bounds
            y_end = min(y1 + y_offset + new_h, canvas_height)
            x_end = min(x1 + x_offset + new_w, canvas_width)
            
            if y1 + y_offset < canvas_height and x1 + x_offset < canvas_width:
                canvas[y1 + y_offset:y_end, x1 + x_offset:x_end] = resized[:y_end - (y1 + y_offset), :x_end - (x1 + x_offset)]
            
            # Add label
            label = region.replace('_', ' ').title()
            label_y = min(y1 + target_h + 15, canvas_height - 5)
            label_x = max(5, x1 + target_w // 2 - len(label) * 4)
            if label_y < canvas_height - 5:
                cv2.putText(canvas, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    
    cv2.imwrite(output_path, canvas)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--img-dir', required=True, help='Directory containing dog images')
    parser.add_argument('--output-dir', default='inference_with_regions', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of images to process')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize facial region cropper
    cropper = DogFacialRegionCropper()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    
    # Get random images
    print(f"Sampling {args.num_samples} images from {args.img_dir}")
    image_paths = get_random_images(args.img_dir, args.num_samples)
    
    # Process each image
    results = []
    print(f"\nProcessing {len(image_paths)} images...")
    
    for idx, img_path in enumerate(tqdm(image_paths)):
        try:
            # Get bbox (simple version - use whole image)
            bbox, img = detect_dog_bbox(img_path)
            h, w = img.shape[:2]
            
            # Create bbox in xyxy format [x1, y1, x2, y2] as expected by MMPose
            bbox_xyxy = np.array([[0, 0, w, h]])  # [x1, y1, x2, y2]
            
            # Run inference
            pose_results = inference_topdown(model, img_path, bboxes=bbox_xyxy)
            
            if len(pose_results) > 0:
                # Get keypoints from first detection
                pred_instances = pose_results[0].pred_instances
                keypoints = pred_instances.keypoints
                scores = pred_instances.keypoint_scores
                
                # Convert to numpy if tensors
                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                
                # Handle shapes properly (remove batch dimension if present)
                if keypoints.ndim == 3:
                    keypoints = keypoints[0]  # [46, 2]
                if scores.ndim == 2:
                    scores = scores[0]  # [46]
                
                # Combine into [46, 3] format
                keypoints_with_conf = np.concatenate([
                    keypoints, scores.reshape(-1, 1)
                ], axis=1)
                
                # Create output folder for this image
                img_name = Path(img_path).stem
                dog_id = Path(img_path).parent.name
                img_folder = os.path.join(args.output_dir, f'{idx+1:04d}_{dog_id}_{img_name}')
                os.makedirs(img_folder, exist_ok=True)
                
                # Save original image
                cv2.imwrite(os.path.join(img_folder, 'original.jpg'), img)
                
                # Save visualization with landmarks
                visualize_landmarks(img, keypoints_with_conf, 
                                  os.path.join(img_folder, 'landmarks.jpg'))
                
                # Save image with region bounding boxes
                cropper.visualize_regions(img, keypoints_with_conf,
                                        os.path.join(img_folder, 'regions_bbox.jpg'),
                                        padding=0.1)
                
                # Crop and save all facial regions
                crops = cropper.crop_all_regions(img, keypoints_with_conf, padding=0.1)
                
                # Create regions subfolder
                regions_folder = os.path.join(img_folder, 'regions')
                os.makedirs(regions_folder, exist_ok=True)
                
                region_info = []
                for region_name, crop_img in crops.items():
                    region_path = os.path.join(regions_folder, f'{region_name}.jpg')
                    cv2.imwrite(region_path, crop_img)
                    h_crop, w_crop = crop_img.shape[:2]
                    region_info.append({
                        'region': region_name,
                        'width': w_crop,
                        'height': h_crop,
                        'path': f'regions/{region_name}.jpg'
                    })
                
                # Create composite view of regions
                if crops:
                    create_composite_regions(crops, os.path.join(img_folder, 'regions_composite.jpg'))
                
                # Save prediction data
                result_data = {
                    'image_path': img_path,
                    'dog_id': dog_id,
                    'image_name': img_name,
                    'output_folder': img_folder,
                    'keypoints': keypoints.tolist(),
                    'scores': scores.tolist(),
                    'avg_confidence': float(scores.mean()),
                    'regions': region_info,
                    'total_landmarks': len(keypoints),
                    'visible_landmarks': int(np.sum(scores > 0.3))
                }
                
                # Save individual result JSON
                with open(os.path.join(img_folder, 'results.json'), 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                results.append(result_data)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Save all predictions to main JSON
    output_json = os.path.join(args.output_dir, 'all_predictions.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary HTML
    create_summary_html(args.output_dir, results)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Inference Complete!")
    print(f"{'='*50}")
    print(f"Processed: {len(results)} / {len(image_paths)} images")
    print(f"Output saved to: {args.output_dir}")
    print(f"All predictions saved to: {output_json}")
    
    if results:
        avg_conf = np.mean([r['avg_confidence'] for r in results])
        print(f"Average confidence: {avg_conf:.3f}")
    
    print(f"\nEach image folder contains:")
    print("  - original.jpg: Original image")
    print("  - landmarks.jpg: Image with numbered landmarks")
    print("  - regions_bbox.jpg: Image with region bounding boxes")
    print("  - regions_composite.jpg: All regions in one view")
    print("  - regions/: Folder with individual cropped regions")
    print("  - results.json: Detailed results for this image")
    
    # Show some sample predictions
    print(f"\nSample predictions (first 3):")
    for r in results[:3]:
        print(f"  {r['image_name']}: conf={r['avg_confidence']:.3f}, regions={len(r['regions'])}")


def create_summary_html(output_dir, results):
    """Create an HTML summary for easy browsing."""
    num_results = len(results)
    avg_confidence = np.mean([r['avg_confidence'] for r in results]) if results else 0
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dog Face Inference Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; text-align: center; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ 
            background: white; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card img {{ 
            width: 100%; 
            height: auto; 
            border-radius: 4px; 
            margin-bottom: 10px;
        }}
        .card h3 {{ margin: 10px 0; color: #333; }}
        .stats {{ font-size: 14px; color: #666; }}
        .links {{ margin-top: 10px; }}
        .links a {{ 
            color: #0066cc; 
            text-decoration: none; 
            margin-right: 15px;
        }}
        .links a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Dog Face Landmark Detection & Region Extraction</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total images processed: <strong>{num_results}</strong></p>
        <p>Average landmark confidence: <strong>{avg_confidence:.3f}</strong></p>
        <p>Facial regions extracted: left_eye, right_eye, nose, mouth, left_ear, right_ear, forehead</p>
    </div>
    
    <div class="grid">
"""
    
    for i, result in enumerate(results):
        folder_name = os.path.basename(result['output_folder'])
        html_content += f"""
        <div class="card">
            <h3>{i+1}. {result['image_name']}</h3>
            <img src="{folder_name}/regions_composite.jpg" alt="Regions">
            <div class="stats">
                <p>Confidence: {result['avg_confidence']:.3f}</p>
                <p>Visible landmarks: {result['visible_landmarks']}/{result['total_landmarks']}</p>
                <p>Regions: {len(result['regions'])}</p>
            </div>
            <div class="links">
                <a href="{folder_name}/landmarks.jpg">Landmarks</a>
                <a href="{folder_name}/regions_bbox.jpg">Bounding Boxes</a>
                <a href="{folder_name}/regions/">Cropped Regions</a>
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(html_content)


if __name__ == '__main__':
    main()
