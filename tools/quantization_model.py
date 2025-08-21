# tools/quantization_model.py
import os
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    quantize_static
)

# --- Paths ---
FP32_MODEL_PATH = "output_inference/rtdetrv3_r18vd_6x.onnx"
INT8_MODEL_PATH = "output_inference/rtdetrv3_r18vd_6x_int8.onnx"
CALIBRATION_FOLDER = "dataset/calibration_640"  # contains npy files

# --- Calibration Data Reader (Your implementation is correct, no changes needed) ---
class NpyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        if not self.files:
            raise ValueError(f"No .npy files found in calibration folder: {folder}")
        self.current_index = 0
        self.sess = onnxruntime.InferenceSession(FP32_MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def get_next(self):
        if self.current_index >= len(self.files):
            return None

        f = self.files[self.current_index]
        self.current_index += 1

        loaded = np.load(os.path.join(self.folder, f), allow_pickle=True)

        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            data_dict = loaded.item()
        elif isinstance(loaded, dict):
            data_dict = loaded
        else:
            raise ValueError(f"Unexpected data format in {f}: expected dict, got {type(loaded)}")

        # Ensure the dictionary is in the correct order for the model inputs
        # Although ONNX Runtime matches by name, providing the correctly ordered dict is good practice.
        return {name: data_dict[name] for name in self.input_names}

    def rewind(self):
        self.current_index = 0

# --- Find nodes to exclude from quantization ---
def get_nodes_to_exclude(model_path):
    """
    Finds nodes that are sensitive to quantization and should be excluded.
    """
    model = onnx.load(model_path)
    nodes_to_exclude = []

    # This list contains operator types that are often problematic for quantization
    # Start with a smaller list and add more if you still see issues.
    sensitive_ops = [
        'Softmax', 'LayerNormalization', 'Gelu',
        'InstanceNormalization', 'Sigmoid', 'Div', 'Pow'
    ]

    for node in model.graph.node:
        if node.op_type in sensitive_ops:
            nodes_to_exclude.append(node.name)

    print(f"Found {len(nodes_to_exclude)} nodes to exclude from quantization.")
    print(f"Excluding nodes: {nodes_to_exclude}")
    return nodes_to_exclude

# --- Run Quantization ---
def run_quantization():
    print("Starting static INT8 quantization...")

    # 1. Initialize the calibration data reader
    calibration_data_reader = NpyCalibrationDataReader(CALIBRATION_FOLDER)

    # 2. Get the list of nodes to exclude
    nodes_to_exclude = get_nodes_to_exclude(FP32_MODEL_PATH)

    # 3. Run quantization
    quantize_static(
        model_input=FP32_MODEL_PATH,
        model_output=INT8_MODEL_PATH,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,      # QDQ is generally better for compatibility
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,  # <-- The crucial new parameter
        per_channel=True  # <-- ✨ THE CRUCIAL NEW PARAMETER
    )
    print(f"✅ INT8 quantized model saved to: {INT8_MODEL_PATH}")

if __name__ == "__main__":
    run_quantization()