# tools/user_script.py

import os
from pathlib import Path
import cv2
import numpy as np

def preprocess(image_path, target_size=(640, 640)):
    """
    Loads and preprocesses a single image for the RT-DETR model.
    This function is adapted from your verify_onnx_model.py script.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Create shape and scale factor tensors
    orig_shape_tensor = np.array([[img.shape[0], img.shape[1]]], dtype=np.float32)

    # Preprocess the image tensor
    resized_img = cv2.resize(img, target_size)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    image_tensor = resized_img.astype(np.float32).transpose(2, 0, 1)[None, :, :, :] / 255.0

    return image_tensor, orig_shape_tensor


def create_calibration_dataloader(data_dir, batch_size, *args, **kwargs):
    """
    This is the dataloader function that Olive will call.
    It yields batches of preprocessed data for model calibration.
    """
    image_filepaths = sorted([p for p in Path(data_dir).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    # NOTE: The input names must match your ONNX model's input names exactly.
    # Use Netron to verify them. These are common names for RT-DETR.
    image_input_name = "images"
    shape_input_name = "orig_target_sizes"

    if batch_size != 1:
        raise ValueError("This dataloader is designed for batch_size=1.")

    for image_path in image_filepaths:
        # Preprocess the image to get the two required tensors
        image_tensor, orig_shape_tensor = preprocess(image_path)

        if image_tensor is not None:
            # The dataloader must yield a tuple of (input_data, label)
            # For calibration, the label is not used, so it can be None.
            # The input_data is a dictionary mapping input names to numpy arrays.
            input_dict = {
                image_input_name: image_tensor,
                shape_input_name: orig_shape_tensor,
            }
            yield input_dict, None