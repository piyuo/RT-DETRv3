import onnx
from onnxconverter_common import float16

# --- Configuration ---
FP32_MODEL_PATH = "output_inference/rtdetrv3_r18vd_6x.onnx"
FP16_MODEL_PATH = "output_inference/rtdetrv3_r18vd_6x_fp16.onnx"

# --- Load the FP32 ONNX model ---
print(f"Loading model from {FP32_MODEL_PATH}...")
model = onnx.load(FP32_MODEL_PATH)
print("Model loaded successfully.")

# --- Define operators to keep in FP32 ---
ops_to_keep_in_fp32 = [
    'Softmax',
    'LayerNormalization',
    'Add',
    'Mul',
    'Div',
    'ReduceMean',
    'Cast' # <--- IMPORTANT ADDITION: Ensure Cast nodes are not converted
]

# --- Convert the model to FP16 ---
print(f"Converting model to FP16. Keeping {ops_to_keep_in_fp32} in FP32...")
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
    op_block_list=ops_to_keep_in_fp32
)
print("Model conversion complete.")

# --- Save the FP16 model ---
onnx.save(model_fp16, FP16_MODEL_PATH)
print(f"✅ FP16 ONNX model saved successfully to {FP16_MODEL_PATH}")