#!/bin/bash
# Script to evaluate your trained dog face landmark model

# First, let's find what checkpoints are available
echo "=========================================="
echo "Finding available checkpoints..."
echo "=========================================="
echo ""

echo "Checkpoints in mmpose_dogflw:"
ls -lh work_dirs/mmpose_dogflw/best*.pth 2>/dev/null || echo "  No best checkpoints found"
ls -lh work_dirs/mmpose_dogflw/epoch*.pth 2>/dev/null || echo "  No epoch checkpoints found"

echo ""
echo "Checkpoints in td-hm_hrnet-w32_udp_dogflw-256x256:"
ls -lh work_dirs/td-hm_hrnet-w32_udp_dogflw-256x256/best*.pth 2>/dev/null || echo "  No best checkpoints found"
ls -lh work_dirs/td-hm_hrnet-w32_udp_dogflw-256x256/epoch*.pth 2>/dev/null || echo "  No epoch checkpoints found"

echo ""
echo "=========================================="
echo "Running Evaluation..."
echo "=========================================="
echo ""

# Set the config and checkpoint paths
CONFIG="configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py"
OUTPUT_DIR="work_dirs/evaluation_results_$(date +%Y%m%d_%H%M%S)"

# Try to find the best checkpoint
if [ -f "work_dirs/mmpose_dogflw/best_NME_epoch_100.pth" ]; then
    CHECKPOINT="work_dirs/mmpose_dogflw/best_NME_epoch_100.pth"
    echo "Using checkpoint: $CHECKPOINT"
elif ls work_dirs/mmpose_dogflw/best_NME_*.pth 1> /dev/null 2>&1; then
    CHECKPOINT=$(ls -t work_dirs/mmpose_dogflw/best_NME_*.pth | head -1)
    echo "Using latest best checkpoint: $CHECKPOINT"
elif ls work_dirs/td-hm_hrnet-w32_udp_dogflw-256x256/best_NME_*.pth 1> /dev/null 2>&1; then
    CHECKPOINT=$(ls -t work_dirs/td-hm_hrnet-w32_udp_dogflw-256x256/best_NME_*.pth | head -1)
    echo "Using checkpoint: $CHECKPOINT"
elif ls work_dirs/mmpose_dogflw/epoch_*.pth 1> /dev/null 2>&1; then
    CHECKPOINT=$(ls -t work_dirs/mmpose_dogflw/epoch_*.pth | tail -1)
    echo "Using latest epoch checkpoint: $CHECKPOINT"
else
    echo "ERROR: No checkpoint found!"
    exit 1
fi

echo ""
echo "Configuration: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Run the evaluation
python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --work-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/*.json"
echo ""


