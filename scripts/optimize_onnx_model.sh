# scripts/optimize_onnx_model.sh

#!/bin/bash

# Activate the environment
source onnx_env/bin/activate

# Optimize the ONNX model
#python3 tools/optimize_onnx_model.py \
#	--model output/rtdetrv3_r18vd_6x.onnx \
#	--out output/olive_rtdetrv3_int8 \
#	--device cpu

#olive auto-opt \
#  --model_name_or_path output/rtdetrv3_r18vd_6x.onnx \
#  --output_path output/olive_rtdetrv3_optimized \
#  --device cpu \
#  --provider CPUExecutionProvider \
#  --precision int8 \
#  --save_config_file \
#  --log_level 1



olive  run --config optimize_onnx_model.json

scripts/verify_onnx_model.sh