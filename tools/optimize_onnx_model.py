# tools/optimize_onnx_model.py

from olive.workflows import run as olive_run
from pathlib import Path

# 1. Define the paths
model_path = "output/rtdetrv3_r18vd_6x.onnx"
calibration_data_path = "calib/output"
output_dir = Path("output/olive_rtdetrv3_static_int8")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Define the configuration for the data loader
calibration_data_config = {
    "user_script": "tools/user_script.py",
    "data_component_config": {
        "type": "CalibrationDataLoader",
        "params": {
            "data_dir": calibration_data_path,
        }
    }
}

# 3. Define the main Olive configuration
olive_config = {
    "input_model": {
        "type": "ONNXModel",
        "config": {
            "model_path": model_path,
        },
    },
    "passes": {
            "type": "OnnxStaticQuantization",
            "config": {
                "quant_format": "QDQ",          # Quantization format: QDQ (Quantize/Dequantize nodes)
                "quant_mode": "static",
                "providers": ["CPUExecutionProvider"],
                "target_device": "cpu",
                "data_config": data_config      # Inline the dict, or provide a path to .json if preferred
            }
    },
    "output_dir": str(output_dir),
}

if __name__ == "__main__":
    print("Starting Olive static quantization workflow...")
    olive_run(olive_config)
    print(f"\n✅ Done. Statically quantized model saved to: {output_dir}")