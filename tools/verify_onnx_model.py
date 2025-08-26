# tools/verify_onnx_model.py
# Verify the exported ONNX model by running inference and visualizing results.

import onnxruntime as ort
import numpy as np
import cv2
import os
import random

# ------------------------------------------------------------
# COCO class labels (80 classes)
# ------------------------------------------------------------
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Pre-generate distinct colors for classes
np.random.seed(42)
CLASS_COLORS = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(COCO_CLASSES))}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def preprocess_image(image_path, target_size=(640, 640)):
    """Load and preprocess image for ONNX model."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    im_shape = np.array([[img.shape[0], img.shape[1]]], dtype=np.float32)
    scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

    resized = cv2.resize(img, target_size)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = resized.astype(np.float32).transpose(2, 0, 1)[None, :, :, :] / 255.0

    return img, resized, im_shape, scale_factor


def normalize_shape(shape):
    """Convert ONNX dynamic dims to -1 for easier comparison."""
    return [(-1 if (isinstance(dim, str) or dim is None) else dim) for dim in shape]


def decode_outputs(outputs, orig_shape, conf_threshold=0.5):
    """Handle both [N,6] and [B,N,C] style outputs."""
    detections = []

    for out in outputs:
        out = np.array(out)

        # Case 1: [N,6] → [class_id, conf, x1,y1,x2,y2]
        if len(out.shape) == 2 and out.shape[1] == 6:
            for det in out:
                cls_id, conf, x1, y1, x2, y2 = det
                if conf >= conf_threshold:
                    detections.append((int(cls_id), float(conf), [x1, y1, x2, y2]))

        # Case 2: [B,N,C] → need to reshape
        elif len(out.shape) == 3 and out.shape[2] >= 6:
            for det in out.reshape(-1, out.shape[-1]):
                cls_id, conf, x1, y1, x2, y2 = det[:6]
                if conf >= conf_threshold:
                    detections.append((int(cls_id), float(conf), [x1, y1, x2, y2]))

    return detections


def draw_detections(image, detections, save_path, class_names=None):
    """Draw bounding boxes with unique color per object and save to file."""
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    for idx, (cls_id, conf, (x1, y1, x2, y2)) in enumerate(detections):
        # Generate a unique color for each object
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))

        # Label: class name if available, otherwise id
        if class_names and 0 <= cls_id < len(class_names):
            label_text = f"{class_names[cls_id]} {conf:.2f}"
        else:
            label_text = f"{cls_id}:{conf:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label_text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
    print(f"Saved result image with detections to {save_path}")

# ------------------------------------------------------------
# Main verification
# ------------------------------------------------------------
def verify_model(onnx_model, image_path, output_path, conf_threshold=0.5):
    # Load model
    sess = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

    # Get input/output info
    input_nodes = sess.get_inputs()
    input_names = [n.name for n in input_nodes]

    # Preprocess image
    orig_img, image_tensor, im_shape_tensor, scale_factor_tensor = preprocess_image(image_path)

    # Match inputs
    input_feed = {}
    for node in input_nodes:
        norm_shape = normalize_shape(node.shape)

        if len(norm_shape) == 4 and norm_shape[1] == 3:
            input_feed[node.name] = image_tensor
        elif len(norm_shape) == 2 and norm_shape[1] == 2:
            if "shape" in node.name:
                input_feed[node.name] = im_shape_tensor
            elif "scale" in node.name:
                input_feed[node.name] = scale_factor_tensor
            else:
                if input_names.index(node.name) == 0:
                    input_feed[node.name] = im_shape_tensor
                else:
                    input_feed[node.name] = scale_factor_tensor
        else:
            raise RuntimeError(f"Unhandled input: {node.name} with shape {node.shape}")

    # Run inference
    outputs = sess.run(None, input_feed)

    # Decode detections
    detections = decode_outputs(outputs, orig_img.shape, conf_threshold)

    print(f"\n--- Verification Result ---")
    print(f"Detected {len(detections)} objects with confidence > {conf_threshold}")
    for i, (cls_id, conf, bbox) in enumerate(detections, 1):
        class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
        print(f"  - Obj {i}: Class={class_name} ({cls_id}), Conf={conf:.3f}, BBox={bbox}")

    # Draw and save
    draw_detections(orig_img.copy(), detections, output_path)


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    verify_model(
        #onnx_model="output/rtdetrv3_r18vd_6x_optimized.onnx",
        onnx_model="output/olive/model/rtdetrv3_r18vd_6x.onnx",
        image_path="demo/demo.jpg",
        output_path="output/demo.jpg",
        conf_threshold=0.5
    )
