"""DogFLW Dataset for PyTorch."""
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DogFLWDataset(Dataset):
    """DogFLW dataset in COCO format."""
    
    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        input_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (128, 128),
        sigma: float = 3.0,
        transform=None,
        mode: str = 'train'
    ):
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform
        self.mode = mode
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id -> image_info and annotation mappings
        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = {}
        for ann in coco_data['annotations']:
            self.annotations[ann['image_id']] = ann
        
        self.image_ids = list(self.annotations.keys())
        self.num_keypoints = coco_data['categories'][0].get('keypoints', None)
        if self.num_keypoints:
            self.num_keypoints = len(self.num_keypoints)
        else:
            # Infer from first annotation
            first_ann = coco_data['annotations'][0]
            self.num_keypoints = len(first_ann['keypoints']) // 3
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        ann = self.annotations[image_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get keypoints and bbox
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)  # Nx3 (x, y, visibility)
        bbox = ann['bbox']  # [x, y, w, h]
        
        # Crop and resize based on bbox
        x, y, w, h = bbox
        center = np.array([x + w / 2, y + h / 2])
        scale = max(w, h) * 1.25  # Add padding
        
        # Crop image
        x1 = max(0, int(center[0] - scale / 2))
        y1 = max(0, int(center[1] - scale / 2))
        x2 = min(image.shape[1], int(center[0] + scale / 2))
        y2 = min(image.shape[0], int(center[1] + scale / 2))
        
        cropped = image[y1:y2, x1:x2]
        
        # Adjust keypoints to cropped coordinates
        keypoints[:, 0] -= x1
        keypoints[:, 1] -= y1
        
        # Resize image and keypoints
        h_cropped, w_cropped = cropped.shape[:2]
        resized = cv2.resize(cropped, self.input_size)
        
        scale_x = self.input_size[0] / w_cropped
        scale_y = self.input_size[1] / h_cropped
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        
        # Generate heatmaps
        heatmaps = self._generate_heatmaps(keypoints)
        
        # Convert to torch tensors
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        heatmaps_tensor = torch.from_numpy(heatmaps).float()
        keypoints_tensor = torch.from_numpy(keypoints).float()
        
        # Normalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return {
            'image': image_tensor,
            'heatmaps': heatmaps_tensor,
            'keypoints': keypoints_tensor,
            'image_id': image_id
        }
    
    def _generate_heatmaps(self, keypoints: np.ndarray) -> np.ndarray:
        """Generate Gaussian heatmaps for keypoints."""
        num_joints = len(keypoints)
        heatmaps = np.zeros((num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        
        # Scale keypoints to heatmap size
        scale_x = self.heatmap_size[0] / self.input_size[0]
        scale_y = self.heatmap_size[1] / self.input_size[1]
        
        for idx, (x, y, vis) in enumerate(keypoints):
            if vis == 0:  # Not visible
                continue
            
            # Scale to heatmap coordinates
            mu_x = x * scale_x
            mu_y = y * scale_y
            
            # Check if keypoint is in bounds
            if mu_x < 0 or mu_y < 0 or mu_x >= self.heatmap_size[0] or mu_y >= self.heatmap_size[1]:
                continue
            
            # Generate Gaussian heatmap
            x_grid = np.arange(0, self.heatmap_size[0])
            y_grid = np.arange(0, self.heatmap_size[1])
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            heatmap = np.exp(-((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / (2 * self.sigma ** 2))
            heatmaps[idx] = heatmap
        
        return heatmaps

