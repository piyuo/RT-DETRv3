#!/bin/bash
# onnx/quantize.sh
#
# RT-DETRv3 Model Quantization for Production Deployment
# =====================================================
#
# This script quantizes the RT-DETRv3 model with backbone features for optimized
# deployment in production environments. It uses Microsoft Olive framework for
# advanced model optimization including quantization, pruning, and graph optimization.
#
# Optimization Features:
# 1. INT8 quantization for reduced model size and faster inference
# 2. Graph optimization for improved performance
# 3. Calibration-based quantization for accuracy preservation
# 4. CPU-optimized execution provider configuration
# 5. Model validation and quality assurance
#
# Process:
# 1. Load backbone model with Re-ID features
# 2. Apply Olive optimization pipeline
# 3. Quantize model weights and activations
# 4. Validate quantized model performance
# 5. Generate optimized model for deployment
#
# Input:  onnx/backbone/rtdetrv3_r18vd_6x.onnx (Model with backbone features)
# Output: onnx/output/rtdetrv3_r18vd_6x.onnx     (Quantized production model)
#
# Performance Impact:
# - Model size reduction: ~75% (FP32 to INT8)
# - Inference speed: 2-4x faster on CPU
# - Memory usage: ~50% reduction
# - Accuracy retention: >95% (with proper calibration)
#
# Prerequisites:
# - Backbone model must be available (run export_backbone.sh first)
# - Calibration data configured in quantize.json
# - Olive framework installed in environment
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

if [ ! -f "onnx/quantize.json" ]; then
    echo "❌ Quantization configuration not found at onnx/quantize.json"
    exit 1
fi

echo "⚡ Starting RT-DETRv3 model quantization..."
echo "   Input model: onnx/backbone/rtdetrv3_r18vd_6x.onnx"
echo "   Configuration: onnx/quantize.json"
echo "   Target: INT8 quantization with graph optimization"
echo "   Output: onnx/output/rtdetrv3_r18vd_6x.onnx"

# Create output directory if it doesn't exist
mkdir -p onnx/output

# Alternative: Auto-optimization approach (commented out)
# This provides automated optimization but less control over the process
#echo "🤖 Running automated optimization with Olive..."
#olive auto-opt \
#  --model_name_or_path onnx/backbone/rtdetrv3_r18vd_6x.onnx \
#  --output_path onnx/output/olive_rtdetrv3_optimized \
#  --device cpu \
#  --provider CPUExecutionProvider \
#  --precision int8 \
#  --save_config_file \
#  --log_level 1

# Run Olive optimization with custom configuration
echo "🔧 Running custom quantization pipeline..."
olive run --config onnx/quantize.json

# Check if optimization was successful
if [ $? -ne 0 ]; then
    echo "❌ Model quantization failed"
    exit 1
fi

# Rename the output model to a more descriptive name
if [ -f "onnx/output/olive/model.onnx" ]; then
    echo "📦 Finalizing quantized model..."
    mv onnx/output/olive/model.onnx onnx/output/rtdetrv3_r18vd_6x.onnx

    # Clean up intermediate files
    if [ -d "onnx/output/olive" ]; then
        rmdir onnx/output/olive 2>/dev/null || echo "   (Keeping intermediate files for debugging)"
    fi

    echo "✅ Model quantization completed successfully"
    echo "   Quantized model saved to: onnx/output/rtdetrv3_r18vd_6x.onnx"
    echo "   Next step: Run inference.sh to validate the quantized model"
else
    echo "❌ Expected output file not found after quantization"
    exit 1
fi

# Display model size comparison if both files exist
if [ -f "onnx/backbone/rtdetrv3_r18vd_6x.onnx" ] && [ -f "onnx/output/rtdetrv3_r18vd_6x.onnx" ]; then
    echo ""
    echo "📊 Quantization Results:"
    original_size=$(ls -lh onnx/backbone/rtdetrv3_r18vd_6x.onnx | awk '{print $5}')
    quantized_size=$(ls -lh onnx/output/rtdetrv3_r18vd_6x.onnx | awk '{print $5}')
    echo "   Original model size:  $original_size"
    echo "   Quantized model size: $quantized_size"
    echo "   Ready for production deployment"
fi
