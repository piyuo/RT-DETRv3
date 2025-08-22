# scripts/init_env.sh
# This script initializes the environment for the project
#!/bin/bash

# PaddlePaddle need Python 3.11 now
brew install python@3.11

# Create an environment with Python 3.11
python3.11 -m venv paddle_env

# Activate the environment
source paddle_env/bin/activate


# Install the compatible PaddlePaddle version
pip install paddlepaddle

# Now, install PaddleSlim and PaddleDetection
pip install opencv-python==4.6.0.66
pip install numpy==1.26.4
pip install paddleslim
pip install paddledet
pip install imgaug