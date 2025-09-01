#!/bin/bash
# onnx/reid_embeddings.sh
#
# Re-ID Embeddings Generator for RT-DETRv3
# ========================================
#
# This script generates Re-ID embeddings from input images using the RT-DETRv3 model
# with extracted backbone features. It provides a complete pipeline for embedding
# generation with comprehensive debugging and validation output.
#
# Features:
# 1. Robust image preprocessing with letterbox support
# 2. Object detection with confidence filtering
# 3. Backbone feature extraction for detected objects
# 4. L2-normalized embedding generation (512-dimensional)
# 5. Quality validation and metrics
# 6. Comprehensive debugging output
# 7. Visualization and analysis tools
#
# Processing Pipeline:
# 1. Load and preprocess input image (letterbox or resize)
# 2. Run RT-DETRv3 inference for object detection
# 3. Extract backbone features for each detected object
# 4. Generate normalized Re-ID embeddings
# 5. Validate embedding quality and consistency
# 6. Save results with detailed metadata
#
# Output: Re-ID embeddings saved in output/reid/ with visualization
#
# Use Cases:
# - Multi-object tracking systems
# - Object re-identification across cameras
# - Surveillance and monitoring applications
# - Research and development of tracking algorithms
#

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source onnx/env/bin/activate

# Verify environment activation
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Verify required files exist
if [ ! -f "onnx/backbone/rtdetrv3_r18vd_6x.onnx" ]; then
    echo "❌ Backbone model not found. Please run export_backbone.sh first."
    exit 1
fi

if [ ! -f "onnx/demo/demo.jpg" ]; then
    echo "❌ Demo image not found at onnx/demo/demo.jpg"
    exit 1
fi

echo "🎯 Starting Re-ID embedding generation..."
echo "   Model: onnx/backbone/rtdetrv3_r18vd_6x.onnx"
echo "   Input image: onnx/demo/demo.jpg"
echo "   Output directory: output/reid/"
echo "   Preprocessing: Letterbox (preserves aspect ratio)"
echo "   Debug mode: Enabled (comprehensive output)"

# Create output directory if it doesn't exist
mkdir -p output/reid

# Generate Re-ID embeddings with letterbox preprocessing and comprehensive debugging
python3 onnx/reid_embeddings.py \
    --model onnx/backbone/rtdetrv3_r18vd_6x.onnx \
    --image onnx/demo/demo.jpg \
    --output output/reid \
    --use-letterbox \
    --debug

# Check generation results
if [ $? -eq 0 ]; then
    echo "✅ Re-ID embedding generation completed successfully"
    echo "   Embeddings and metadata saved to: output/reid/"
    echo "   Check the debug output above for quality metrics"
else
    echo "❌ Re-ID embedding generation failed"
    echo "   Check the output above for specific errors"
    exit 1
fi

echo ""
echo "📊 Generation Summary:"
echo "   • Object detection: Completed"
echo "   • Feature extraction: Successful"
echo "   • Embedding generation: Validated"
echo "   • Quality metrics: Calculated"
echo "   • Ready for tracking applications"
