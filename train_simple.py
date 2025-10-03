"""Simple PyTorch trainer for DogFLW."""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from dataset import DogFLWDataset


class HRNetPoseModel(nn.Module):
    """HRNet-based pose estimation model."""
    
    def __init__(self, num_keypoints=46, pretrained=True):
        super().__init__()
        # Load HRNet-W32 from timm
        self.backbone = timm.create_model(
            'hrnet_w32',
            pretrained=pretrained,
            features_only=True,
            out_indices=[0]  # Get the highest resolution features
        )
        
        # HRNet-W32 outputs 64 channels at H/2 resolution
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )
        
    def forward(self, x):
        # Get features from HRNet
        features = self.backbone(x)[0]  # [B, 64, H/2, W/2]
        
        # Generate heatmaps
        heatmaps = self.head(features)  # [B, num_keypoints, H/2, W/2]
        
        return heatmaps


class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps with visibility weighting."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_heatmaps, target_heatmaps, keypoints):
        """
        Args:
            pred_heatmaps: [B, K, H, W]
            target_heatmaps: [B, K, H, W]
            keypoints: [B, K, 3] with visibility in [:, :, 2]
        """
        batch_size, num_joints = pred_heatmaps.shape[:2]
        
        # Create visibility mask
        vis_mask = keypoints[:, :, 2].unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        vis_mask = vis_mask.expand_as(pred_heatmaps)
        
        # Compute MSE loss only for visible joints
        loss = F.mse_loss(pred_heatmaps * vis_mask, target_heatmaps * vis_mask, reduction='sum')
        loss = loss / (vis_mask.sum() + 1e-6)
        
        return loss


def calculate_nme(pred_heatmaps, keypoints, normalize='box'):
    """Calculate Normalized Mean Error."""
    batch_size, num_joints, h, w = pred_heatmaps.shape
    
    # Get predicted coordinates from heatmaps (argmax)
    pred_heatmaps_flat = pred_heatmaps.view(batch_size, num_joints, -1)
    max_vals, max_indices = pred_heatmaps_flat.max(dim=2)
    
    pred_x = (max_indices % w).float()
    pred_y = (max_indices // w).float()
    
    # Scale back to input image size (256x256)
    scale_x = 256.0 / w
    scale_y = 256.0 / h
    pred_x *= scale_x
    pred_y *= scale_y
    
    # Get ground truth coordinates
    gt_coords = keypoints[:, :, :2]  # [B, K, 2]
    vis = keypoints[:, :, 2]  # [B, K]
    
    # Compute distances
    pred_coords = torch.stack([pred_x, pred_y], dim=2)  # [B, K, 2]
    distances = torch.norm(pred_coords - gt_coords, dim=2)  # [B, K]
    
    # Apply visibility mask
    valid_distances = distances * vis
    num_valid = vis.sum()
    
    if num_valid == 0:
        return 0.0
    
    # Normalize by image size (simple box normalization)
    if normalize == 'box':
        norm_factor = 256.0  # diagonal of 256x256
    else:
        norm_factor = 1.0
    
    nme = (valid_distances.sum() / num_valid) / norm_factor
    return nme.item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_nme = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device)
        target_heatmaps = batch['heatmaps'].to(device)
        keypoints = batch['keypoints'].to(device)
        
        # Forward pass
        pred_heatmaps = model(images)
        loss = criterion(pred_heatmaps, target_heatmaps, keypoints)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            nme = calculate_nme(pred_heatmaps, keypoints)
        
        total_loss += loss.item()
        total_nme += nme
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'nme': f'{nme:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    avg_nme = total_nme / len(dataloader)
    
    return avg_loss, avg_nme


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_nme = 0
    
    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        images = batch['image'].to(device)
        target_heatmaps = batch['heatmaps'].to(device)
        keypoints = batch['keypoints'].to(device)
        
        # Forward pass
        pred_heatmaps = model(images)
        loss = criterion(pred_heatmaps, target_heatmaps, keypoints)
        
        # Calculate metrics
        nme = calculate_nme(pred_heatmaps, keypoints)
        
        total_loss += loss.item()
        total_nme += nme
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'nme': f'{nme:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    avg_nme = total_nme / len(dataloader)
    
    return avg_loss, avg_nme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, 
                       default='/Users/michael/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW')
    parser.add_argument('--ann-root', type=str,
                       default='/Users/michael/Projects/760face/data/dogflw/annotations')
    parser.add_argument('--work-dir', type=str, default='work_dirs/simple_hrnet')
    parser.add_argument('--epochs', type=int, default=210)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Create work directory
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {args.device}")
    
    # Create datasets
    train_dataset = DogFLWDataset(
        ann_file=os.path.join(args.ann_root, 'train.json'),
        img_dir=os.path.join(args.data_root, 'train/images'),
        mode='train'
    )
    
    val_dataset = DogFLWDataset(
        ann_file=os.path.join(args.ann_root, 'val.json'),
        img_dir=os.path.join(args.data_root, 'test/images'),
        mode='val'
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = HRNetPoseModel(num_keypoints=46, pretrained=True)
    model = model.to(args.device)
    
    # Loss and optimizer
    criterion = JointsMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_nme = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_nme = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch)
        print(f"Train Loss: {train_loss:.4f}, Train NME: {train_nme:.4f}")
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_loss, val_nme = validate(model, val_loader, criterion, args.device)
            print(f"Val Loss: {val_loss:.4f}, Val NME: {val_nme:.4f}")
            
            # Save best model
            if val_nme < best_nme:
                best_nme = val_nme
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_nme': val_nme,
                }, os.path.join(args.work_dir, 'best_model.pth'))
                print(f"✅ Saved best model (NME: {val_nme:.4f})")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.work_dir, f'epoch_{epoch}.pth'))
        
        scheduler.step()
    
    print(f"\n{'='*50}")
    print(f"Training complete! Best NME: {best_nme:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

