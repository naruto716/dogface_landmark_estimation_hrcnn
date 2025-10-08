#!/bin/bash
# Extract landmarks and bboxes to JSON for training pipeline

echo "🐕 Extracting landmarks and bboxes to JSON..."
echo ""

# Run extraction
python extract_landmarks_json.py \
    --config configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py \
    --checkpoint work_dirs/mmpose_dogflw/best_NME_epoch_100.pth \
    --img-dir /path/to/your/dataset \
    --output-dir landmarks_json \
    --device cuda:0

echo ""
echo "✅ Done! JSON files saved to landmarks_json/"
echo ""
echo "Each JSON file contains:"
echo "  - landmarks: 46 keypoints (x, y, confidence)"
echo "  - region_bboxes: Bounding boxes for facial regions"
echo "  - Image metadata"
