"""Visualize landmark patterns to identify facial regions."""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_predictions(json_path):
    """Load predictions from inference results."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_landmark_patterns(predictions, output_dir):
    """Analyze and visualize landmark patterns across multiple images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all landmarks normalized to [0, 1]
    all_landmarks = []
    
    for pred in predictions[:50]:  # Use first 50 predictions
        if pred['avg_confidence'] > 0.5:  # Only use confident predictions
            keypoints = np.array(pred['keypoints'])
            scores = np.array(pred['scores'])
            
            # Get image dimensions (assumes 256x256 from your config)
            img_size = 256
            
            # Normalize keypoints to [0, 1]
            normalized_kps = keypoints / img_size
            
            # Filter by confidence
            mask = scores > 0.3
            if mask.sum() > 40:  # At least 40 good landmarks
                all_landmarks.append(normalized_kps[mask])
    
    if not all_landmarks:
        print("No confident predictions found!")
        return
    
    # Calculate mean positions for each landmark
    mean_landmarks = np.zeros((46, 2))
    landmark_counts = np.zeros(46)
    
    for landmarks in all_landmarks:
        for i in range(min(len(landmarks), 46)):
            if not np.isnan(landmarks[i]).any():
                mean_landmarks[i] += landmarks[i]
                landmark_counts[i] += 1
    
    # Avoid division by zero
    mask = landmark_counts > 0
    mean_landmarks[mask] /= landmark_counts[mask].reshape(-1, 1)
    
    # Visualize mean landmark positions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: All landmarks with numbers
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0)  # Flip Y axis for image coordinates
    ax1.set_aspect('equal')
    ax1.set_title('Mean Landmark Positions (All 46 points)', fontsize=14)
    
    colors = plt.cm.tab20(np.linspace(0, 1, 46))
    
    for i in range(46):
        if landmark_counts[i] > 0:
            x, y = mean_landmarks[i]
            ax1.scatter(x, y, c=[colors[i]], s=100, edgecolors='black', linewidth=1)
            ax1.text(x + 0.01, y + 0.01, str(i), fontsize=8, fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Normalized X')
    ax1.set_ylabel('Normalized Y')
    
    # Plot 2: Grouped by likely facial regions (based on spatial clustering)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1, 0)
    ax2.set_aspect('equal')
    ax2.set_title('Likely Facial Regions (Based on Spatial Clustering)', fontsize=14)
    
    # Define regions based on typical positions
    # These are educated guesses - you'll need to refine based on visual inspection
    regions = {
        'left_eye': [],
        'right_eye': [],
        'nose': [],
        'mouth': [],
        'left_ear': [],
        'right_ear': [],
        'face_contour': []
    }
    
    # Classify landmarks based on position (rough estimates)
    for i in range(46):
        if landmark_counts[i] > 0:
            x, y = mean_landmarks[i]
            
            # Eye region (upper face, left and right)
            if 0.2 < y < 0.45:
                if x < 0.4:
                    regions['left_eye'].append(i)
                elif x > 0.6:
                    regions['right_eye'].append(i)
            
            # Nose region (center, middle)
            elif 0.4 < y < 0.6 and 0.4 < x < 0.6:
                regions['nose'].append(i)
            
            # Mouth region (lower center)
            elif y > 0.6 and 0.3 < x < 0.7:
                regions['mouth'].append(i)
            
            # Ear regions (far left/right, upper)
            elif y < 0.4:
                if x < 0.2:
                    regions['left_ear'].append(i)
                elif x > 0.8:
                    regions['right_ear'].append(i)
            
            # Face contour (edges)
            else:
                regions['face_contour'].append(i)
    
    # Plot regions with different colors
    region_colors = {
        'left_eye': 'blue',
        'right_eye': 'cyan',
        'nose': 'red',
        'mouth': 'orange',
        'left_ear': 'green',
        'right_ear': 'lime',
        'face_contour': 'purple'
    }
    
    for region_name, indices in regions.items():
        if indices:
            points = mean_landmarks[indices]
            valid = landmark_counts[indices] > 0
            if valid.any():
                ax2.scatter(points[valid, 0], points[valid, 1], 
                           c=region_colors[region_name], s=100, 
                           label=f'{region_name} ({len(indices)} pts)',
                           edgecolors='black', linewidth=1)
                
                # Add landmark numbers
                for idx, i in enumerate(indices):
                    if valid[idx]:
                        x, y = points[idx]
                        ax2.text(x + 0.01, y + 0.01, str(i), fontsize=7)
    
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax2.set_xlabel('Normalized X')
    ax2.set_ylabel('Normalized Y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'landmark_pattern_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"Saved landmark pattern analysis to {output_dir}/landmark_pattern_analysis.png")
    
    # Save region mapping to JSON
    region_mapping = {
        region: indices for region, indices in regions.items() if indices
    }
    
    with open(os.path.join(output_dir, 'landmark_regions.json'), 'w') as f:
        json.dump(region_mapping, f, indent=2)
    print(f"Saved landmark region mapping to {output_dir}/landmark_regions.json")
    
    # Print summary
    print("\nLandmark Region Summary:")
    for region, indices in regions.items():
        if indices:
            print(f"  {region}: landmarks {indices}")
    
    return regions


def visualize_on_sample_images(predictions, regions, output_dir, num_samples=5):
    """Visualize regions on sample images."""
    sample_dir = os.path.join(output_dir, 'sample_regions')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Select diverse samples
    samples = sorted(predictions, key=lambda x: x['avg_confidence'], reverse=True)[:num_samples]
    
    for idx, pred in enumerate(samples):
        img_path = pred['image_path']
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        keypoints = np.array(pred['keypoints'])
        scores = np.array(pred['scores'])
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Original image with all landmarks
        axes[0].imshow(img_rgb)
        axes[0].set_title('All Landmarks')
        for i, (x, y) in enumerate(keypoints):
            if scores[i] > 0.3:
                axes[0].plot(x, y, 'ro', markersize=3)
        axes[0].axis('off')
        
        # Show each region
        region_colors = {
            'left_eye': 'blue',
            'right_eye': 'cyan', 
            'nose': 'red',
            'mouth': 'orange',
            'left_ear': 'green',
            'right_ear': 'lime',
            'face_contour': 'purple'
        }
        
        for ax_idx, (region_name, indices) in enumerate(regions.items(), 1):
            if ax_idx >= len(axes):
                break
                
            axes[ax_idx].imshow(img_rgb)
            axes[ax_idx].set_title(region_name.replace('_', ' ').title())
            
            if indices:
                # Plot landmarks for this region
                region_points = []
                for i in indices:
                    if i < len(keypoints) and scores[i] > 0.3:
                        x, y = keypoints[i]
                        axes[ax_idx].plot(x, y, 'o', color=region_colors[region_name], markersize=5)
                        region_points.append([x, y])
                
                # Draw bounding box if we have points
                if len(region_points) >= 2:
                    region_points = np.array(region_points)
                    x_min, y_min = region_points.min(axis=0)
                    x_max, y_max = region_points.max(axis=0)
                    
                    # Add padding
                    padding = 0.1
                    width = x_max - x_min
                    height = y_max - y_min
                    x_min -= width * padding
                    x_max += width * padding
                    y_min -= height * padding
                    y_max += height * padding
                    
                    # Draw rectangle
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor=region_colors[region_name], 
                                   facecolor='none', linestyle='--')
                    axes[ax_idx].add_patch(rect)
            
            axes[ax_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'sample_{idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_samples} sample visualizations to {sample_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze landmark patterns from inference results')
    parser.add_argument('--predictions', required=True, help='Path to predictions.json from inference')
    parser.add_argument('--output-dir', default='landmark_analysis', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of sample images to visualize')
    args = parser.parse_args()
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")
    
    # Analyze landmark patterns
    regions = analyze_landmark_patterns(predictions, args.output_dir)
    
    # Visualize on sample images
    if regions:
        visualize_on_sample_images(predictions, regions, args.output_dir, args.num_samples)
    
    print(f"\n✅ Analysis complete! Check {args.output_dir}/ for results")


if __name__ == '__main__':
    main()
