#!/usr/bin/env python3
"""
Quick script to test a trained model and get metrics.
Simpler than the full MMPose test pipeline.
"""
import argparse
import json
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Quick test a pose model')
    parser.add_argument('--config', 
                       default='configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py',
                       help='test config file path')
    parser.add_argument('--checkpoint',
                       help='checkpoint file (if not provided, will search for best checkpoint)')
    parser.add_argument('--work-dir',
                       help='output directory')
    args = parser.parse_args()
    
    # Find checkpoint if not provided
    if not args.checkpoint:
        print("Searching for checkpoints...")
        
        # Search paths in order of preference
        search_paths = [
            'work_dirs/mmpose_dogflw/best_NME_epoch_100.pth',
            'work_dirs/mmpose_dogflw/best_NME_*.pth',
            'work_dirs/td-hm_hrnet-w32_udp_dogflw-256x256/best_NME_*.pth',
            'work_dirs/mmpose_dogflw/epoch_*.pth',
        ]
        
        checkpoint = None
        for pattern in search_paths:
            if '*' in pattern:
                # Use glob
                matches = sorted(Path('.').glob(pattern))
                if matches:
                    checkpoint = str(matches[-1])  # Use latest
                    break
            else:
                # Direct path
                if Path(pattern).exists():
                    checkpoint = pattern
                    break
        
        if not checkpoint:
            print("ERROR: No checkpoint found! Please specify with --checkpoint")
            return
        
        print(f"Found checkpoint: {checkpoint}")
        args.checkpoint = checkpoint
    
    # Set work dir
    if not args.work_dir:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.work_dir = f'work_dirs/evaluation_{timestamp}'
    
    print("\n" + "="*70)
    print("🧪 EVALUATING MODEL")
    print("="*70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.work_dir}")
    print("="*70 + "\n")
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    
    # Build the runner from config
    runner = Runner.from_cfg(cfg)
    
    # Run evaluation
    print("\nRunning evaluation on test set...\n")
    metrics = runner.test()
    
    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.6f}")
            else:
                print(f"{key:30s}: {value}")
    
    # Save results
    results_file = Path(args.work_dir) / 'evaluation_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Results saved to: {results_file}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()


