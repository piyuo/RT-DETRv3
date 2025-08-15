# scripts/export_onnx.sh
# Export the model to ONNX format

#!/bin/bash
python3 tools/export_model.py \
    -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
    -o weights=weights/rtdetrv3_r18vd_6x.pdparams \
    export_onnx=True

paddle2onnx \
    --model_dir output_inference/rtdetrv3_r18vd_6x_coco \
    --model_filename model.json \
    --params_filename model.pdiparams \
    --save_file output_inference/rtdetrv3_r18vd_6x.onnx \
    --opset_version 11 \
    --enable_onnx_checker True




#What should you do now?
#Test the ONNX model inference with a compatible ONNX runtime that supports IR version 11 or higher (e.g., latest onnxruntime):
#pip install --upgrade onnxruntime
#If you want to reduce warnings, install these optional packages:
#pip install colored onnx_graphsurgeon
#If you want to try enabling partitioning or other export flags for better folding results, you can check paddle2onnx docs, but these are optional.