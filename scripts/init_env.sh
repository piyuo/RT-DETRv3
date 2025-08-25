# scripts/init_env.sh
# This script initializes the environment for the project
#!/bin/bash


# Create an environment with Python 3.11
python3 -m venv onnx_env

# Activate the environment
source onnx_env/bin/activate


# Install onnxruntime-tools if not already installed
pip install numpy==1.26.4 --upgrade
pip install --upgrade olive-ai onnxruntime onnx onnxruntime-tools pillow
pip install opencv-python==4.9.0.80