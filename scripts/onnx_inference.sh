# scripts/onnx_inference.sh

#!/bin/bash

# Activate the environment
source onnx_env/bin/activate

# run inference on demo.jpg
python3 scripts/onnx_inference.py --debug

# rename output model name
mv output/olive/model.onnx output/rtdetrv3_r18vd_6x.onnx