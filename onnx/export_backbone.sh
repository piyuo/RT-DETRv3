# onnx/export_backbone.sh
# Auto-detect and export best backbone feature (C4)

# Activate the environment
source onnx/env/bin/activate

#!/bin/bash

python3 onnx/export_backbone.py \
    --input onnx/input/rtdetrv3_r18vd_6x.onnx \
    --output onnx/backbone/rtdetrv3_r18vd_6x.onnx \
    --level C3
    #--list-only
