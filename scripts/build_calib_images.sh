# scripts/build_calib_images.sh

#!/bin/bash

# Activate the environment
source onnx_env/bin/activate

# convert calib/input images to 640x640 and save to calib/output
python3 tools/build_calib_images.py