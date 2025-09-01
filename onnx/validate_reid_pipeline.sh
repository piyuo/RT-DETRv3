#!/bin/bash
# onnx/validate_reid_pipeline.sh
#
# Re-ID Pipeline Comprehensive Validator
# =====================================
#
# This script performs thorough validation of the Re-ID embedding pipeline to ensure
# robust operation in production environments. It validates model structure, feature
# extraction, embedding quality, and provides detailed performance metrics.
#
# Validation Components:
# 1. Model architecture verification
# 2. Feature map quality assessment
# 3. Detection format validation
# 4. Coordinate transformation accuracy
# 5. ROI extraction correctness
# 6. Embedding separability analysis
# 7. Quality metrics evaluation
#
# Prerequisites:
# - Backbone feature model must exist (run export_backbone.sh first)
# - Demo image must be available for testing
#
# Output: Comprehensive validation report in JSON format
#
# Important: Check export_backbone.sh output for the correct feature map name
# Example output: "Exported feature: Concat.3" -> use --feature-map-name Concat.3
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

echo "🔍 Starting comprehensive Re-ID pipeline validation..."
echo "   Model: onnx/backbone/rtdetrv3_r18vd_6x.onnx"
echo "   Test image: onnx/demo/demo.jpg"
echo "   Feature map: Concat.3 (C4 layer for RT-DETRv3-R18vd-6x)"
echo "   Output: onnx/validation/reid_report.json"

# Create validation output directory if it doesn't exist
mkdir -p onnx/validation

# Run comprehensive validation with detailed reporting
python3 onnx/validate_reid_pipeline.py \
    --model onnx/backbone/rtdetrv3_r18vd_6x.onnx \
    --image onnx/demo/demo.jpg \
    --feature-map-name Concat.3 \
    --output onnx/validation/reid_report.json

# Check validation results
if [ $? -eq 0 ]; then
    echo "✅ Re-ID pipeline validation completed successfully"
    echo "   Validation report saved to: onnx/validation/reid_report.json"
    echo "   Next step: Run botsort_integration_test.sh to test tracking integration"
else
    echo "❌ Re-ID pipeline validation failed"
    echo "   Check the output above for specific error details"
    exit 1
fi

# Display validation summary if report exists
if [ -f "onnx/validation/reid_report.json" ]; then
    echo ""
    echo "📊 Validation Summary:"
    echo "   Check the JSON report for detailed metrics and recommendations"
fi
