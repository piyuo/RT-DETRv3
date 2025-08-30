# scripts/export_model.sh
# Export the model from weights in  /weights

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
