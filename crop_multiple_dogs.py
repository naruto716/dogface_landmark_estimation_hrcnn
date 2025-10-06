"""Process multiple dogs and save each dog's facial regions in separate folders."""
import os
import cv2
import numpy as np
import json
from pathlib import Path
from config import TRAIN_IMAGES, TRAIN_ANN
from crop_facial_regions import DogFacialRegionCropper


def process_multiple_dogs(num_dogs=10):
    """Process multiple dogs and save their facial regions in separate folders."""
    
    # Load annotations
    print("Loading annotations...")
    with open(TRAIN_ANN, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping
    img_to_ann = {ann['image_id']: ann for ann in coco_data['annotations']}
    id_to_img = {img['id']: img for img in coco_data['images']}
    
    # Initialize cropper
    cropper = DogFacialRegionCropper()
    
    # Create main output directory
    output_base = "dog_facial_regions"
    os.makedirs(output_base, exist_ok=True)
    
    # Process dogs
    processed = 0
    
    print(f"\nProcessing {num_dogs} dogs...")
    print("=" * 60)
    
    for img_id, img_info in id_to_img.items():
        if processed >= num_dogs:
            break
            
        if img_id not in img_to_ann:
            continue
        
        # Load image
        img_path = os.path.join(TRAIN_IMAGES, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get annotations
        ann = img_to_ann[img_id]
        
        # Convert keypoints
        keypoints_flat = ann['keypoints']
        keypoints = []
        for i in range(0, len(keypoints_flat), 3):
            x, y, v = keypoints_flat[i:i+3]
            conf = 1.0 if v > 0 else 0.0
            keypoints.append([x, y, conf])
        keypoints = np.array(keypoints)
        
        # Create folder for this dog
        dog_name = Path(img_info['file_name']).stem
        dog_folder = os.path.join(output_base, f"dog_{processed+1:03d}_{dog_name}")
        os.makedirs(dog_folder, exist_ok=True)
        
        # Save original image
        cv2.imwrite(os.path.join(dog_folder, "original.jpg"), img)
        
        # Save original with bounding boxes
        cropper.visualize_regions(img, keypoints, 
                                 os.path.join(dog_folder, "original_with_regions.jpg"),
                                 padding=0.1)
        
        # Crop all regions
        crops = cropper.crop_all_regions(img, keypoints, padding=0.1)
        
        # Save individual crops
        print(f"\nDog {processed + 1}: {dog_name}")
        print("-" * 40)
        
        crop_info = []
        for region_name, crop_img in crops.items():
            output_path = os.path.join(dog_folder, f"{region_name}.jpg")
            cv2.imwrite(output_path, crop_img)
            h, w = crop_img.shape[:2]
            print(f"  ✓ {region_name:12s} - {w}x{h} pixels")
            crop_info.append({
                'region': region_name,
                'width': w,
                'height': h,
                'path': f"{region_name}.jpg"
            })
        
        # Create a composite image showing all regions
        create_composite_view(crops, os.path.join(dog_folder, "all_regions.jpg"))
        
        # Save metadata
        metadata = {
            'image_name': img_info['file_name'],
            'image_id': img_id,
            'dog_folder': dog_folder,
            'regions': crop_info,
            'total_landmarks': len(keypoints),
            'visible_landmarks': int(np.sum(keypoints[:, 2] > 0))
        }
        
        with open(os.path.join(dog_folder, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        processed += 1
    
    # Create summary
    create_summary(output_base, processed)
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed {processed} dogs!")
    print(f"📁 Results saved in: {output_base}/")
    print(f"\nEach dog folder contains:")
    print("  - original.jpg: Original image")
    print("  - original_with_regions.jpg: Image with region bounding boxes")
    print("  - [region].jpg: Individual cropped regions")
    print("  - all_regions.jpg: Composite view of all regions")
    print("  - metadata.json: Information about the crops")


def create_composite_view(crops, output_path):
    """Create a nice composite view of all facial regions."""
    # Calculate grid size
    cell_size = 150
    padding = 10
    
    # Create canvas - fixed dimensions
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


def create_summary(output_base, num_processed):
    """Create a summary HTML file for easy browsing."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dog Facial Regions Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .dog-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .dog-card {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; }}
        .dog-card img {{ width: 100%; height: auto; border-radius: 4px; }}
        .dog-card h3 {{ margin: 10px 0; }}
        .region-list {{ font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <h1>Dog Facial Regions - {num_processed} Dogs Processed</h1>
    <div class="dog-grid">
"""
    
    # Add each dog
    for i in range(num_processed):
        dog_folders = sorted([f for f in os.listdir(output_base) if f.startswith(f"dog_{i+1:03d}_")])
        if dog_folders:
            dog_folder = dog_folders[0]
            dog_name = dog_folder.split('_', 2)[2]
            
            html_content += f"""
        <div class="dog-card">
            <h3>Dog {i+1}: {dog_name}</h3>
            <img src="{dog_folder}/all_regions.jpg" alt="Dog {i+1} regions">
            <p class="region-list">
                <a href="{dog_folder}/">View all files</a> |
                Regions: left_eye, right_eye, nose, mouth, ears, forehead
            </p>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(os.path.join(output_base, "index.html"), 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-dogs', type=int, default=10, 
                       help='Number of dogs to process')
    args = parser.parse_args()
    
    process_multiple_dogs(args.num_dogs)
