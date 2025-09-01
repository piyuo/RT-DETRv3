# scripts/onnx_quantize.sh

#!/bin/bash

# Activate the environment
source onnx/env/bin/activate


#olive auto-opt \
#  --model_name_or_path output/rtdetrv3_r18vd_6x.onnx \
#  --output_path output/olive_rtdetrv3_optimized \
#  --device cpu \
#  --provider CPUExecutionProvider \
#  --precision int8 \
#  --save_config_file \
#  --log_level 1

# use olive to quantize the onnx model
olive  run --config onnx/quantize.json

# Rename the output model to a more descriptive name
mv -f onnx/output/olive/model.onnx onnx/output/rtdetrv3_r18vd_6x.onnx

# run inference on demo.jpg
python3 onnx/inference.py --debug