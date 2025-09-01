# onnx/botsort_integration_test.sh
# demonstrates how to use the generated Re-ID embeddings with BoT-SORT tracking

# Activate the environment
source onnx/env/bin/activate

#!/bin/bash
# check the output of export_backbone.sh, find the Exported feature: *****, and use it as --feature-map-name
# Example:  --feature-map-name Concat.5 // C3 layer for RT-DETRv3-R18vd-6x

python3 onnx/botsort_integration_test.py