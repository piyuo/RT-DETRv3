# onnx/validate_reid_pipeline.sh
# Run comprehensive validation of the REID pipeline

# Activate the environment
source onnx/env/bin/activate

#!/bin/bash

python3 onnx/validate_reid_pipeline.py \
    --model onnx/backbone/rtdetrv3_r18vd_6x.onnx \
    --image onnx/demo/demo.jpg \
    --output onnx/validation/reid_report.json
