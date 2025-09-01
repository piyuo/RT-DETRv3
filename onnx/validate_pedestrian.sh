# onnx/validate_pedestrian.sh
# test person-to-person distinguishability

# Activate the environment
source onnx/env/bin/activate

#!/bin/bash
# check the output of export_backbone.sh, find the Exported feature: *****, and use it as --feature-map-name
# Example:  --feature-map-name Concat.5 // C3 layer for RT-DETRv3-R18vd-6x

python3 onnx/validate_pedestrian.py \
    --model onnx/backbone/rtdetrv3_r18vd_6x.onnx \
    --image onnx/demo/demo.jpg \
    --feature-map-name Concat.3
