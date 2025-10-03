"""Check what the trained model is actually predicting."""
import sys
import os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import DogFLWDataset
from train_simple import HRNetPoseModel, calculate_nme

# Get paths
data_root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW')
ann_root = sys.argv[2] if len(sys.argv) > 2 else 'data/dogflw/annotations'
checkpoint = sys.argv[3] if len(sys.argv) > 3 else 'work_dirs/dogflw_hrnet_lr1e4/best_model.pth'

print(f"Loading checkpoint: {checkpoint}")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HRNetPoseModel(num_keypoints=46, pretrained=False)
checkpoint_data = torch.load(checkpoint, map_location=device)
model.load_state_dict(checkpoint_data['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Model loaded. Epoch: {checkpoint_data.get('epoch', 'unknown')}")
print(f"Val NME from checkpoint: {checkpoint_data.get('val_nme', 'unknown')}")

# Load dataset
dataset = DogFLWDataset(
    ann_file=os.path.join(ann_root, 'val.json'),
    img_dir=os.path.join(data_root, 'test/images')
)

# Test on first sample
sample = dataset[0]
image_tensor = sample['image'].unsqueeze(0).to(device)
gt_heatmaps = sample['heatmaps'].numpy()
keypoints = sample['keypoints'].numpy()

# Get model prediction
with torch.no_grad():
    pred_heatmaps = model(image_tensor)
    pred_heatmaps_np = pred_heatmaps[0].cpu().numpy()
    
    # Calculate NME
    nme = calculate_nme(pred_heatmaps, sample['keypoints'].unsqueeze(0).to(device))

print(f"\nPrediction NME: {nme:.4f}")
print(f"Pred heatmaps shape: {pred_heatmaps_np.shape}")
print(f"Pred heatmaps value range: [{pred_heatmaps_np.min():.3f}, {pred_heatmaps_np.max():.3f}]")
print(f"GT heatmaps value range: [{gt_heatmaps.min():.3f}, {gt_heatmaps.max():.3f}]")

# Visualize comparison for first 6 keypoints
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i in range(6):
    # Ground truth heatmap
    axes[i//2, (i%2)*2].imshow(gt_heatmaps[i], cmap='hot', vmin=0, vmax=1)
    gt_peak_idx = gt_heatmaps[i].argmax()
    gt_peak_y, gt_peak_x = np.unravel_index(gt_peak_idx, gt_heatmaps[i].shape)
    axes[i//2, (i%2)*2].plot(gt_peak_x, gt_peak_y, 'g+', markersize=20, markeredgewidth=3)
    axes[i//2, (i%2)*2].set_title(f'GT Heatmap {i}\nPeak: ({gt_peak_x}, {gt_peak_y})\nMax: {gt_heatmaps[i].max():.3f}')
    axes[i//2, (i%2)*2].axis('off')
    
    # Predicted heatmap
    axes[i//2, (i%2)*2+1].imshow(pred_heatmaps_np[i], cmap='hot', vmin=0, vmax=1)
    pred_peak_idx = pred_heatmaps_np[i].argmax()
    pred_peak_y, pred_peak_x = np.unravel_index(pred_peak_idx, pred_heatmaps_np[i].shape)
    axes[i//2, (i%2)*2+1].plot(pred_peak_x, pred_peak_y, 'c+', markersize=20, markeredgewidth=3)
    
    # Calculate pixel distance between peaks
    peak_dist = np.sqrt((pred_peak_x - gt_peak_x)**2 + (pred_peak_y - gt_peak_y)**2)
    
    axes[i//2, (i%2)*2+1].set_title(f'PRED Heatmap {i}\nPeak: ({pred_peak_x}, {pred_peak_y})\nMax: {pred_heatmaps_np[i].max():.3f}\nDist: {peak_dist:.1f}px')
    axes[i//2, (i%2)*2+1].axis('off')

plt.tight_layout()
plt.savefig('debug_model_predictions.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization to debug_model_predictions.png")

# Calculate average peak distance
distances = []
for i in range(min(46, pred_heatmaps_np.shape[0])):
    if keypoints[i, 2] > 0:  # visible
        gt_peak_idx = gt_heatmaps[i].argmax()
        gt_peak_y, gt_peak_x = np.unravel_index(gt_peak_idx, gt_heatmaps[i].shape)
        
        pred_peak_idx = pred_heatmaps_np[i].argmax()
        pred_peak_y, pred_peak_x = np.unravel_index(pred_peak_idx, pred_heatmaps_np[i].shape)
        
        dist = np.sqrt((pred_peak_x - gt_peak_x)**2 + (pred_peak_y - gt_peak_y)**2)
        distances.append(dist)

print(f"\n📊 Statistics across all keypoints:")
print(f"  Average peak distance: {np.mean(distances):.1f} px (on 128x128 heatmap)")
print(f"  Scaled to image (256x256): {np.mean(distances) * 2:.1f} px")
print(f"  This explains NME: {np.mean(distances) * 2 / 256:.3f}")

