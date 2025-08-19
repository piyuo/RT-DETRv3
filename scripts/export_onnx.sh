# scripts/export_onnx.sh
# Export the model to ONNX format

#export POLYGRAPHY_AUTOINSTALL_DEPS=1
# python3 -m pip install -U pip
# pip install onnx==1.16.2 onnxruntime==1.22.1 polygraphy --force-reinstall
# pip install numpy==1.26.4 --only-binary=:all: --force-reinstall
# pip install numba==0.56.4

#!/bin/bash
python3 tools/export_model.py \
    -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
    -o weights=weights/rtdetrv3_r18vd_6x.pdparams \
    export_onnx=True

# compile paddle2onnx on macOS x64 , cause paddle2onnx only support arm64 for now, and re-compile if libprotobuf version is changed.
# it seems RT-DETRv3 only support opset_version 16, so we set it to 16 here.
paddle2onnx \
    --model_dir output_inference/rtdetrv3_r18vd_6x_coco \
    --model_filename model.json \
    --params_filename model.pdiparams \
    --save_file output_inference/rtdetrv3_r18vd_6x.onnx \
    --opset_version 16 \
    --enable_auto_update_opset False \
    --optimize_tool None \
    --enable_onnx_checker True
    #--enable_verbose True

python3 tools/fix_model.py
# replace the original model with fixed model
mv -f output_inference/rtdetrv3_r18vd_6x_fixed.onnx output_inference/rtdetrv3_r18vd_6x.onnx
python3 tools/load_model.py


#python -m onnxsim output_inference/rtdetrv3_r18vd_6x.onnx output_inference/rtdetrv3_r18vd_6x_simplified.onnx --no-large-tensor
