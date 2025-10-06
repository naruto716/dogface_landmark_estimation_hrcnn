"""Utilities for cropping specific facial regions based on landmarks."""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import json


class DogFacialRegionCropper:
    """Crop specific facial regions from dog faces using detected landmarks."""
    
    def __init__(self, landmark_regions: Optional[Dict[str, List[int]]] = None):
        """
        Initialize with landmark region mapping.
        
        Args:
            landmark_regions: Dict mapping region names to landmark indices.
                             If None, uses default mapping (to be refined based on analysis).
        """
        if landmark_regions is None:
            # Correct mapping based on manual landmark analysis
            self.landmark_regions = {
                'left_eye': [17, 19, 21, 23],
                'right_eye': [16, 18, 20, 22],
                'nose': [25, 26, 27, 32, 33, 34, 35],
                'mouth': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45],
                'left_ear': [1, 3, 5, 7, 9, 11, 13],
                'right_ear': [0, 2, 4, 6, 8, 10, 12],
                'forehead': [14, 15, 0, 1, 20, 21]  # Note: 0, 1, 20, 21 are shared with ears/eyes
            }
        else:
            self.landmark_regions = landmark_regions
    
    def load_regions_from_json(self, json_path: str):
        """Load landmark region mapping from JSON file."""
        with open(json_path, 'r') as f:
            self.landmark_regions = json.load(f)
    
    def get_region_bbox(self, keypoints: np.ndarray, region: str, 
                       padding: float = 0.2, min_confidence: float = 0.3) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box for a specific facial region.
        
        Args:
            keypoints: Array of shape (N, 3) with (x, y, confidence) for each landmark
            region: Name of the region ('left_eye', 'right_eye', 'nose', 'mouth', etc.)
            padding: Padding factor to add around the bounding box (0.2 = 20% padding)
            min_confidence: Minimum confidence threshold for including a landmark
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None if region not found
        """
        if region not in self.landmark_regions:
            print(f"Warning: Unknown region '{region}'")
            return None
        
        indices = self.landmark_regions[region]
        
        # Get valid landmarks for this region
        valid_points = []
        for idx in indices:
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf >= min_confidence:
                    valid_points.append([x, y])
        
        if len(valid_points) < 2:  # Need at least 2 points to define a box
            return None
        
        valid_points = np.array(valid_points)
        
        # Calculate bounding box
        x_min, y_min = valid_points.min(axis=0)
        x_max, y_max = valid_points.max(axis=0)
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        
        x_min -= width * padding
        x_max += width * padding
        y_min -= height * padding
        y_max += height * padding
        
        # Convert to integers
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def crop_region(self, image: np.ndarray, keypoints: np.ndarray, 
                   region: str, padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop a specific facial region from the image.
        
        Args:
            image: Input image
            keypoints: Array of shape (N, 3) with (x, y, confidence) for each landmark
            region: Name of the region to crop
            padding: Padding factor
            
        Returns:
            Cropped image region or None if region not found
        """
        bbox = self.get_region_bbox(keypoints, region, padding)
        if bbox is None:
            return None
        
        x_min, y_min, x_max, y_max = bbox
        
        # Clip to image bounds
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return None
        
        return image[y_min:y_max, x_min:x_max]
    
    def crop_all_regions(self, image: np.ndarray, keypoints: np.ndarray, 
                        padding: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Crop all defined facial regions from the image.
        
        Args:
            image: Input image
            keypoints: Array of shape (N, 3) with (x, y, confidence) for each landmark
            padding: Padding factor
            
        Returns:
            Dictionary mapping region names to cropped images
        """
        crops = {}
        
        for region in self.landmark_regions.keys():
            cropped = self.crop_region(image, keypoints, region, padding)
            if cropped is not None:
                crops[region] = cropped
        
        return crops
    
    def visualize_regions(self, image: np.ndarray, keypoints: np.ndarray, 
                         output_path: str, padding: float = 0.2):
        """
        Visualize all regions with bounding boxes on the image.
        
        Args:
            image: Input image
            keypoints: Array of shape (N, 3) with (x, y, confidence) for each landmark
            output_path: Path to save the visualization
            padding: Padding factor for bounding boxes
        """
        vis_img = image.copy()
        
        # Define colors for each region
        region_colors = {
            'left_eye': (255, 0, 0),      # Blue
            'right_eye': (255, 255, 0),    # Cyan
            'nose': (0, 0, 255),           # Red
            'mouth': (0, 165, 255),        # Orange
            'left_ear': (0, 255, 0),       # Green
            'right_ear': (0, 255, 127),    # Light green
            'forehead': (128, 0, 128),     # Purple
            'face_contour': (255, 0, 255)  # Magenta
        }
        
        # Draw bounding boxes for each region
        for region, color in region_colors.items():
            bbox = self.get_region_bbox(keypoints, region, padding)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color, 1)  # Thin boundary
                
                # Add label
                label = region.replace('_', ' ').title()
                cv2.putText(vis_img, label, (x_min, y_min - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw landmarks
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        cv2.imwrite(output_path, vis_img)
    
    def get_region_statistics(self, keypoints: np.ndarray) -> Dict[str, Dict]:
        """
        Get statistics about each region (center, size, confidence).
        
        Args:
            keypoints: Array of shape (N, 3) with (x, y, confidence) for each landmark
            
        Returns:
            Dictionary with statistics for each region
        """
        stats = {}
        
        for region, indices in self.landmark_regions.items():
            valid_points = []
            confidences = []
            
            for idx in indices:
                if idx < len(keypoints):
                    x, y, conf = keypoints[idx]
                    if conf > 0.3:
                        valid_points.append([x, y])
                        confidences.append(conf)
            
            if valid_points:
                valid_points = np.array(valid_points)
                center = valid_points.mean(axis=0)
                bbox = self.get_region_bbox(keypoints, region, padding=0)
                
                stats[region] = {
                    'center': center.tolist(),
                    'num_visible': len(valid_points),
                    'avg_confidence': float(np.mean(confidences)),
                    'bbox': bbox,
                    'size': (bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else None
                }
        
        return stats


def create_region_grid(crops: Dict[str, np.ndarray], grid_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Create a grid visualization of all cropped regions.
    
    Args:
        crops: Dictionary mapping region names to cropped images
        grid_size: Size to resize each crop to
        
    Returns:
        Grid image showing all regions
    """
    # Define grid layout
    layout = [
        ['left_ear', 'left_eye', 'right_eye', 'right_ear'],
        ['', 'nose', 'nose', ''],
        ['', 'mouth', 'mouth', '']
    ]
    
    rows = len(layout)
    cols = len(layout[0])
    
    # Create empty grid
    grid = np.ones((rows * grid_size[1], cols * grid_size[0], 3), dtype=np.uint8) * 240
    
    # Place each crop in the grid
    for row_idx, row in enumerate(layout):
        for col_idx, region in enumerate(row):
            if region and region in crops:
                # Resize crop to grid size
                crop = crops[region]
                resized = cv2.resize(crop, grid_size)
                
                # Place in grid
                y_start = row_idx * grid_size[1]
                y_end = y_start + grid_size[1]
                x_start = col_idx * grid_size[0]
                x_end = x_start + grid_size[0]
                
                grid[y_start:y_end, x_start:x_end] = resized
    
    return grid


# Example usage
if __name__ == '__main__':
    # This is just an example - update with your actual data
    print("DogFacialRegionCropper utility loaded.")
    print("Use this class in your inference script to crop facial regions.")
    print("\nExample usage:")
    print("```python")
    print("from crop_facial_regions import DogFacialRegionCropper")
    print("")
    print("# Initialize cropper")
    print("cropper = DogFacialRegionCropper()")
    print("# Or load custom regions from analysis:")
    print("# cropper.load_regions_from_json('landmark_analysis/landmark_regions.json')")
    print("")
    print("# In your inference loop:")
    print("# keypoints_with_conf shape: (46, 3) with (x, y, confidence)")
    print("crops = cropper.crop_all_regions(img, keypoints_with_conf)")
    print("")
    print("# Save individual crops")
    print("for region_name, crop_img in crops.items():")
    print("    cv2.imwrite(f'{region_name}.jpg', crop_img)")
    print("```")
