#!/bin/bash
# Test the facial region groupings on ground truth data

echo "🐕 Testing facial region groupings on DogFLW dataset..."
echo ""

# Run the test script
python test_region_groupings.py \
    --num-samples 20 \
    --output-dir region_grouping_test

echo ""
echo "✅ Done! Check the 'region_grouping_test' folder for results:"
echo "  - comparison_*.png: Side-by-side original vs regions"  
echo "  - regions_*.png: Images with bounding boxes and landmark indices"
echo "  - landmark_index_reference.png: Reference showing all landmark numbers"
echo "  - current_region_mapping.json: Current landmark groupings"
echo ""
echo "Review these images to verify if the landmark groupings are correct!"
