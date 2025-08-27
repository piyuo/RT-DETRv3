# scripts/init_env.sh
# This script initializes the environment for the project
#!/bin/bash


# Create an environment with Python 3.11
python3 -m venv onnx_env

# Activate the environment
source onnx_env/bin/activate


# Install onnxruntime-tools if not already installed
pip install numpy==1.26.4 --upgrade
pip install numba==0.56.4
pip install --upgrade olive-ai onnxruntime onnx onnxruntime-tools pillow onnxconverter-common
pip install opencv-python==4.9.0.80

# Install paddlepaddle
pip install paddlepaddle
pip install scipy
pip install imgaug
pip install scikit-learn

# for rtdetr.py
pip install torchvision
pip install pycocotools
pip install onnxoptimizer
