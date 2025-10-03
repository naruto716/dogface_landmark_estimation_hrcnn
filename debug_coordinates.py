"""Debug script to check coordinate consistency."""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset import DogFLWDataset

# Load one sample
dataset = DogFLWDataset(
    ann_file='data/dogflw/annotations/train.json',
    img_dir='/Users/michael/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW/train/images'
)

sample = dataset[0]
image = sample['image'].numpy().transpose(1, 2, 0)  # CHW -> HWC
image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # denormalize
image = np.clip(image, 0, 1)

keypoints = sample['keypoints'].numpy()
heatmaps = sample['heatmaps'].numpy()

print(f"Image shape: {image.shape}")
print(f"Heatmaps shape: {heatmaps.shape}")
print(f"Keypoints shape: {keypoints.shape}")
print(f"\nKeypoint stats:")
print(f"  X range: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
print(f"  Y range: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
print(f"  Visibility: {keypoints[:, 2].sum():.0f}/{len(keypoints)} visible")

# Check if keypoints are in image bounds
h, w = image.shape[:2]
valid_x = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w)
valid_y = (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)
valid = valid_x & valid_y & (keypoints[:, 2] > 0)
print(f"  Valid keypoints: {valid.sum()}/{len(keypoints)}")

# Visualize first 5 keypoints vs heatmaps
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Show image with keypoints
axes[0, 0].imshow(image)
for i in range(min(5, len(keypoints))):
    x, y, v = keypoints[i]
    if v > 0:
        axes[0, 0].plot(x, y, 'ro', markersize=8)
        axes[0, 0].text(x, y, str(i), color='white', fontsize=8)
axes[0, 0].set_title('Image with GT Keypoints')
axes[0, 0].axis('off')

# Show first 5 heatmaps
for i in range(min(5, len(heatmaps))):
    row = (i + 1) // 3
    col = (i + 1) % 3
    axes[row, col].imshow(heatmaps[i], cmap='hot')
    
    # Find peak in heatmap
    peak_idx = heatmaps[i].argmax()
    peak_y, peak_x = np.unravel_index(peak_idx, heatmaps[i].shape)
    
    # Scale to image coordinates
    scale_x = w / heatmaps.shape[2]
    scale_y = h / heatmaps.shape[1]
    pred_x = peak_x * scale_x
    pred_y = peak_y * scale_y
    
    # Compare with GT
    gt_x, gt_y, v = keypoints[i]
    error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    
    axes[row, col].set_title(f'Heatmap {i}\nGT: ({gt_x:.0f},{gt_y:.0f})\nPeak: ({pred_x:.0f},{pred_y:.0f})\nError: {error:.1f}px')
    axes[row, col].plot(peak_x, peak_y, 'g+', markersize=15, markeredgewidth=2)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('debug_coordinates.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization to debug_coordinates.png")

# Print detailed comparison for first keypoint
print(f"\nDetailed check for keypoint 0:")
kp = keypoints[0]
hm = heatmaps[0]
print(f"  GT keypoint: ({kp[0]:.1f}, {kp[1]:.1f}) vis={kp[2]:.0f}")
print(f"  Heatmap max value: {hm.max():.4f}")
print(f"  Heatmap peak location: {np.unravel_index(hm.argmax(), hm.shape)}")

# Scale heatmap peak to image coords
peak_idx = hm.argmax()
peak_y, peak_x = np.unravel_index(peak_idx, hm.shape)
scale_x = 256.0 / hm.shape[1]
scale_y = 256.0 / hm.shape[0]
pred_x = peak_x * scale_x
pred_y = peak_y * scale_y
print(f"  Predicted location (scaled): ({pred_x:.1f}, {pred_y:.1f})")
print(f"  Error: {np.sqrt((pred_x - kp[0])**2 + (pred_y - kp[1])**2):.1f} pixels")

