#!/bin/bash
# Run inference with facial region extraction

echo "🐕 Running dog face inference with region extraction..."
echo ""

# Run inference on sample images
python inference_with_regions.py \
    --config configs/dogflw/td-hm_hrnet-w32_udp_dogflw-256x256.py \
    --checkpoint checkpoints/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192-73ede547_20220914.pth \
    --img-dir ~/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW/test/images \
    --output-dir inference_with_regions \
    --num-samples 20

echo ""
echo "✅ Done! Check the 'inference_with_regions' folder for results:"
echo "   - Each image has its own folder (e.g., 0001_dogID_imageName/)"
echo "   - Inside each folder:"
echo "     • original.jpg - Original image"
echo "     • landmarks.jpg - Image with numbered landmarks"  
echo "     • regions_bbox.jpg - Image with region bounding boxes"
echo "     • regions_composite.jpg - All regions in grid view"
echo "     • regions/ - Folder with individual cropped regions"
echo "     • results.json - Detailed results"
echo ""
echo "   - index.html - Browse all results in your web browser!"
