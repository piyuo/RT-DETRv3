# onnx/reid_embeddings.sh
# RE-ID Embeddings Generator for RT-DETRv3 with Backbone Features

# Activate the environment
source onnx/env/bin/activate

#!/bin/bash

# With letterbox preprocessing (for models trained with aspect ratio preservation)
python3 onnx/reid_embeddings.py \
    --model onnx/backbone/rtdetrv3_r18vd_6x.onnx \
    --image onnx/demo/demo.jpg \
    --output output/reid \
    --use-letterbox \
    --debug
