# onnx/inference.sh
# Verify the output (possibly quantized) ONNX model by running inference and visualizing results.

# Activate the environment
source onnx/env/bin/activate

#!/bin/bash

# run inference on demo.jpg
python3 onnx/inference.py --debug