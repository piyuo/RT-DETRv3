# scripts/export_model.sh
# Export the model to ONNX format

# Activate the environment
source onnx_env/bin/activate

#!/bin/bash


# export the model using the provided config and weights
python3 tools/export_model.py \
    -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
    -o weights=weights/rtdetrv3_r18vd_6x.pdparams \
    --output_dir output

rm -rf output/paddlepaddle
mv -f output/rtdetrv3_r18vd_6x_coco output/paddlepaddle
rm -rf output/rtdetrv3_r18vd_6x_coco


# compile paddle2onnx on macOS x64 , cause paddle2onnx only support arm64 for now, and re-compile if libprotobuf version is changed.
# it seems RT-DETRv3 only support opset_version 16, so we set it to 16 here.
paddle2onnx \
    --model_dir output/paddlepaddle \
    --model_filename model.json \
    --params_filename model.pdiparams \
    --save_file output/rtdetrv3_r18vd_6x_raw.onnx \
    --opset_version 16 \
    --enable_auto_update_opset False \
    --optimize_tool None \
    --enable_onnx_checker True
    #--enable_verbose True

# paddlepaddle export may have some issues, we need to fix it using onnx_fix_export.py
python3 tools/onnx_fix_export.py
# replace the original model with fixed model
mv -f output/rtdetrv3_r18vd_6x_fixed.onnx output/rtdetrv3_r18vd_6x_raw.onnx

# verify the model can be loaded correctly
python3 tools/onnx_fix_verify.py
