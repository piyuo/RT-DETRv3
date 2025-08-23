# scripts/act.sh
# export and optimize the model with Auto Compression Toolkit (ACT)

#!/bin/bash

# Dataset paths
CALIBRATION_DATASET_PATH="dataset/coco/calibration_dataset"

# 1. Export the model
#echo "Step 1: Exporting the model..."
#python3 tools/export_model.py \
#    -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
#    -o weights=weights/rtdetrv3_r18vd_6x.pdparams \
#    --output_dir output

# 2. Perform model compression with ACT
echo "Step 2: Starting model compression..."
python tools/run.py --config_path=./configs/quant/rtdetr_quant_cfg.yml --save_dir='./output_inference/' --devices='cpu'

#echo "Compression complete. Optimized model saved to output/rtdetrv3_r18vd_6x_quant/"



python3 tools/export_model.py \
    -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
    -o weights=weights/rtdetrv3_r18vd_6x.pdparams \
    --output_dir output
