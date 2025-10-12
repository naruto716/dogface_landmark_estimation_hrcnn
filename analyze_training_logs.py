#!/usr/bin/env python3
"""
Script to analyze MMPose training logs and extract key metrics.
Run this on your server where the actual training logs are located.
"""
import re
import sys
from pathlib import Path
from collections import defaultdict
import json


def parse_training_log(log_file):
    """Parse MMPose training log and extract metrics."""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {log_file}")
    print(f"{'='*70}\n")
    
    # Extract training configuration
    print("📋 TRAINING CONFIGURATION:")
    print("-" * 70)
    
    # Max epochs
    max_epochs_match = re.search(r'max_epochs[\'"]?\s*[:=]\s*(\d+)', content)
    if max_epochs_match:
        max_epochs = max_epochs_match.group(1)
        print(f"Max Epochs: {max_epochs}")
    
    # Learning rate
    lr_match = re.search(r'lr[\'"]?\s*[:=]\s*([\d.e-]+)', content)
    if lr_match:
        lr = lr_match.group(1)
        print(f"Learning Rate: {lr}")
    
    # Batch size
    batch_size_match = re.search(r'batch_size[\'"]?\s*[:=]\s*(\d+)', content)
    if batch_size_match:
        batch_size = batch_size_match.group(1)
        print(f"Batch Size: {batch_size}")
    
    # Checkpoint loaded from
    load_from_match = re.search(r'load_from.*?[\'"](.*?)[\'"]', content)
    if load_from_match:
        load_from = load_from_match.group(1)
        print(f"Loaded from: {load_from}")
    
    # Dataset info
    num_samples_train = re.findall(r'(\d+)\s+samples.*train', content, re.IGNORECASE)
    num_samples_val = re.findall(r'(\d+)\s+samples.*val', content, re.IGNORECASE)
    
    print("\n" + "="*70)
    print("📊 TRAINING PROGRESS:")
    print("-" * 70)
    
    # Find all training iteration logs
    # Pattern: Epoch [1][50/588] lr: 0.0003 loss: 0.0123
    train_pattern = r'Epoch\s+\[(\d+)\]\[(\d+)/(\d+)\].*?loss:\s*([\d.]+)'
    train_matches = re.findall(train_pattern, content)
    
    if train_matches:
        epochs_seen = defaultdict(list)
        for epoch, iter_num, total_iters, loss in train_matches:
            epochs_seen[epoch].append(float(loss))
        
        print(f"\nTotal Training Iterations Found: {len(train_matches)}")
        print(f"Epochs with Training Data: {len(epochs_seen)}")
        
        # Show training loss progression (every 5 epochs or so)
        sorted_epochs = sorted([int(e) for e in epochs_seen.keys()])
        print("\n--- Training Loss by Epoch (avg) ---")
        for epoch in sorted_epochs[::5] if len(sorted_epochs) > 10 else sorted_epochs:
            epoch_str = str(epoch)
            if epoch_str in epochs_seen:
                avg_loss = sum(epochs_seen[epoch_str]) / len(epochs_seen[epoch_str])
                print(f"Epoch {epoch:3d}: Avg Loss = {avg_loss:.6f}")
    else:
        print("⚠️  No training iteration logs found!")
    
    print("\n" + "="*70)
    print("🎯 VALIDATION METRICS:")
    print("-" * 70)
    
    # Find validation NME scores
    # Pattern: val/NME: 0.0123
    val_pattern = r'Epoch\(val\)\s+\[(\d+)\].*?NME:\s*([\d.]+)'
    val_matches = re.findall(val_pattern, content)
    
    if not val_matches:
        # Try alternate pattern
        val_pattern2 = r'Epoch.*?\[(\d+)\].*?val.*?NME[:\s]*([\d.]+)'
        val_matches = re.findall(val_pattern2, content, re.IGNORECASE)
    
    if val_matches:
        print("\n--- Validation NME Scores ---")
        best_nme = float('inf')
        best_epoch = 0
        
        for epoch, nme in val_matches:
            nme_val = float(nme)
            print(f"Epoch {epoch:3d}: NME = {nme_val:.6f}")
            if nme_val < best_nme:
                best_nme = nme_val
                best_epoch = epoch
        
        print(f"\n✅ Best Validation NME: {best_nme:.6f} at Epoch {best_epoch}")
    else:
        print("⚠️  No validation NME scores found!")
        print("   The log might be incomplete or validation hasn't run yet.")
    
    print("\n" + "="*70)
    print("💾 CHECKPOINTS:")
    print("-" * 70)
    
    # Find checkpoint saving logs
    ckpt_pattern = r'(Saving checkpoint|saved.*checkpoint).*?(epoch_\d+|best)'
    ckpt_matches = re.findall(ckpt_pattern, content, re.IGNORECASE)
    
    if ckpt_matches:
        print(f"\nCheckpoints saved: {len(ckpt_matches)}")
        # Show last few checkpoints
        for match in ckpt_matches[-5:]:
            print(f"  - {match[1] if len(match) > 1 else match[0]}")
    else:
        print("No checkpoint information found in log.")
    
    # Check if training completed
    if re.search(r'(Training.*complete|Finished)', content, re.IGNORECASE):
        print("\n✅ Training appears to have completed!")
    elif train_matches:
        last_epoch = max([int(e) for e in epochs_seen.keys()])
        print(f"\n⏳ Last recorded epoch: {last_epoch}")
        if max_epochs_match and int(max_epochs) > last_epoch:
            print(f"   Training may still be in progress or was interrupted.")
    
    print("\n" + "="*70 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_logs.py <path_to_log_file_or_directory>")
        print("\nExamples:")
        print("  python analyze_training_logs.py work_dirs/*/20*.log")
        print("  python analyze_training_logs.py work_dirs/mmpose_dogflw/")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        # Single log file
        parse_training_log(path)
    elif path.is_dir():
        # Directory - find all .log files
        log_files = sorted(path.rglob("*.log"))
        if not log_files:
            print(f"No .log files found in {path}")
            sys.exit(1)
        
        print(f"Found {len(log_files)} log file(s)")
        for log_file in log_files:
            parse_training_log(log_file)
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()


