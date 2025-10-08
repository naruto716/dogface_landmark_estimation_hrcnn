"""
Dog Face Landmark Predictor - Clean inference API
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from mmpose.apis import init_model, inference_topdown
from landmark_regions import DogFacialRegionCropper


class DogFacePredictor:
    """
    Clean API for dog face landmark detection and facial region extraction.
    
    Example:
        >>> predictor = DogFacePredictor('config.py', 'model.pth')
        >>> result = predictor.predict('dog.jpg')
        >>> landmarks = result['landmarks']  # (46, 3) array
        >>> regions = result['regions']      # Dict of cropped images
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
        region_padding: float = 0.1
    ):
        """
        Initialize the dog face predictor.
        
        Args:
            config_path: Path to model config file
            checkpoint_path: Path to trained checkpoint (.pth file)
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
            region_padding: Padding factor for cropped regions (0.1 = 10%)
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.region_padding = region_padding
        
        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = init_model(config_path, checkpoint_path, device=device)
        
        # Initialize region cropper
        self.cropper = DogFacialRegionCropper()
        
        print(f"✅ Model loaded successfully on {device}")
    
    def predict(
        self,
        image_path: str,
        extract_regions: bool = True,
        min_confidence: float = 0.3
    ) -> Dict:
        """
        Run inference on a single dog image.
        
        Args:
            image_path: Path to dog image
            extract_regions: Whether to extract facial regions
            min_confidence: Minimum confidence threshold for landmarks
            
        Returns:
            Dictionary containing:
                - landmarks: (46, 3) array with x, y, confidence
                - regions: Dict of cropped region images (if extract_regions=True)
                - avg_confidence: Average landmark confidence
                - visible_landmarks: Number of visible landmarks
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Create bbox (use whole image)
        bbox_xyxy = np.array([[0, 0, w, h]])
        
        # Run inference
        pose_results = inference_topdown(self.model, image_path, bboxes=bbox_xyxy)
        
        if len(pose_results) == 0:
            raise ValueError("No dog face detected in image")
        
        # Extract keypoints
        pred_instances = pose_results[0].pred_instances
        keypoints = pred_instances.keypoints
        scores = pred_instances.keypoint_scores
        
        # Convert to numpy
        if hasattr(keypoints, 'cpu'):
            keypoints = keypoints.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        
        # Handle batch dimensions
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if scores.ndim == 2:
            scores = scores[0]
        
        # Combine into (46, 3) format
        landmarks = np.concatenate([keypoints, scores.reshape(-1, 1)], axis=1)
        
        # Prepare result
        result = {
            'landmarks': landmarks,
            'avg_confidence': float(scores.mean()),
            'visible_landmarks': int((scores > min_confidence).sum())
        }
        
        # Extract regions if requested
        if extract_regions:
            result['regions'] = self.cropper.crop_all_regions(
                img, landmarks, padding=self.region_padding
            )
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        extract_regions: bool = True
    ) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            extract_regions: Whether to extract facial regions
            
        Returns:
            List of result dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path, extract_regions=extract_regions)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return results
    
    def save_visualization(
        self,
        image_path: str,
        output_dir: str,
        save_regions: bool = True,
        save_landmarks: bool = True,
        save_bbox: bool = True
    ):
        """
        Save visualizations of landmarks and regions.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            save_regions: Save individual cropped regions
            save_landmarks: Save image with numbered landmarks
            save_bbox: Save image with region bounding boxes
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get predictions
        result = self.predict(image_path, extract_regions=save_regions)
        
        # Load original image
        img = cv2.imread(image_path)
        img_name = Path(image_path).stem
        
        # Save landmarks visualization
        if save_landmarks:
            vis_img = img.copy()
            for i, (x, y, conf) in enumerate(result['landmarks']):
                if conf > 0.3:
                    color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
                    cv2.circle(vis_img, (int(x), int(y)), 3, color, -1)
                    cv2.putText(vis_img, str(i), (int(x) + 5, int(y) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imwrite(os.path.join(output_dir, f'{img_name}_landmarks.jpg'), vis_img)
        
        # Save bounding boxes
        if save_bbox:
            self.cropper.visualize_regions(
                img, result['landmarks'],
                os.path.join(output_dir, f'{img_name}_regions_bbox.jpg'),
                padding=self.region_padding
            )
        
        # Save individual regions
        if save_regions and 'regions' in result:
            regions_dir = os.path.join(output_dir, 'regions')
            os.makedirs(regions_dir, exist_ok=True)
            
            for region_name, region_img in result['regions'].items():
                cv2.imwrite(
                    os.path.join(regions_dir, f'{img_name}_{region_name}.jpg'),
                    region_img
                )
        
        print(f"✅ Saved visualizations to {output_dir}")
    
    def get_region_centers(self, landmarks: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Get the center point of each facial region.
        
        Args:
            landmarks: (46, 3) array of landmarks
            
        Returns:
            Dictionary mapping region names to (x, y) center coordinates
        """
        centers = {}
        
        for region_name, indices in self.cropper.landmark_regions.items():
            valid_points = []
            for idx in indices:
                if idx < len(landmarks) and landmarks[idx, 2] > 0.3:
                    valid_points.append(landmarks[idx, :2])
            
            if valid_points:
                center = np.mean(valid_points, axis=0)
                centers[region_name] = tuple(center)
        
        return centers


# Convenience function
def predict_dog_face(
    image_path: str,
    config_path: str,
    checkpoint_path: str,
    device: str = 'cuda:0'
) -> Dict:
    """
    Quick one-liner for inference.
    
    Example:
        >>> result = predict_dog_face('dog.jpg', 'config.py', 'model.pth')
    """
    predictor = DogFacePredictor(config_path, checkpoint_path, device)
    return predictor.predict(image_path)
