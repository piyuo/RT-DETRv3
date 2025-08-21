# tools/make_calibration_640.py

import os
import cv2
import numpy as np

# --- Paths ---
SRC_DIR = "dataset/penn_fudan_ped/train_images"
DST_DIR = "dataset/calibration_640"

# --- Config ---
TARGET_SIZE = 640
# Common normalization for COCO/RT-DETR
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

os.makedirs(DST_DIR, exist_ok=True)

def preprocess_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Failed to load {img_path}")
        return

    h, w = img.shape[:2]

    # Scale to fit inside 640x640
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Pad to 640x640 (top-left padding)
    padded = np.full((TARGET_SIZE, TARGET_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # Convert to float32 + normalize
    img_float = padded.astype(np.float32) / 255.0
    img_norm = (img_float - MEAN) / STD

    # Prepare calibration dict
    calibration_dict = {
        'im_shape': np.array([[new_h, new_w]], dtype=np.float32),       # shape [1,2]
        'image': np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...],    # shape [1,3,640,640]
        'scale_factor': np.array([[scale, scale]], dtype=np.float32)    # shape [1,2]
    }

    # Save as .npy
    np.save(save_path, calibration_dict)

def main():
    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        src_path = os.path.join(SRC_DIR, fname)
        dst_name = os.path.splitext(fname)[0] + ".npy"
        dst_path = os.path.join(DST_DIR, dst_name)

        preprocess_image(src_path, dst_path)
        print(f"✅ Saved {dst_path}")

if __name__ == "__main__":
    main()
