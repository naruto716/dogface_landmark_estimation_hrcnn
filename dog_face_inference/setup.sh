#!/bin/bash
# Setup script for dog face inference package

echo "🐕 Setting up Dog Face Inference Package"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create models directory
echo ""
echo "📁 Creating directories..."
mkdir -p models
mkdir -p output

echo ""
echo "✅ Setup complete!"
echo ""
echo "⚠️  Next steps:"
echo "   1. Copy your trained checkpoint to models/dog_face_model.pth"
echo "   2. Update paths in example.py"
echo "   3. Run: python example.py"
echo ""
echo "📚 See README.md for full documentation"
