import argparse
from pathlib import Path
from olive.workflows import run as olive_run
import json

def make_config(model_path, output_dir, device, do_quant, calib_dir, optimization_level="all"):
    # System: always "LocalSystem"
    systems = {
        "local_system": {"type": "LocalSystem"}
    }

    # Evaluator for metrics and dataloader
    evaluators = {
        "common_evaluator": {
            "metrics": [{
                "name": "latency",
                "type": "latency",
                "sub_type": "avg",
                "user_config": {
                    "user_script": "user_script.py",
                    "data_dir": calib_dir if calib_dir else "",
                    "dataloader_func": "create_calibration_dataloader",
                    "batch_size": 1
                }
            }],
            "target": "local_system"
        }
    }

    # Engine references to defined systems and evaluator
    engine = {
        "host": "local_system",
        "target": "local_system",
        "cache_dir": ".cache",
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "exhaustive"
        },
        "evaluator": "common_evaluator"
    }

    # Passes section
    passes = {
        "ort_opt": {
            "type": "OrtOptimization",
            "config": {
                "optimization_level": optimization_level,
                "target_device": device
            }
        }
    }

    if do_quant:
        if not calib_dir:
            raise ValueError("Quantization requested but --calib-dir not provided.")
        passes["int8_quant"] = {
            "type": "OnnxQuantization",
            "depends": ["ort_opt"],
            "config": {
                "user_script": "user_script.py",
                "dataloader_func": "create_calibration_dataloader",
                "func_args": {
                    "batch_size": 1,
                    "data_dir": calib_dir
                },
                "quant_format": "QDQ",
                "activation_type": "int8",
                "weight_type": "int8",
                "nodes_to_exclude": []
            }
        }

    input_model = {
        "type": "OnnxModel",
        "config": {"model_path": model_path}
    }

    return {
        "input_model": input_model,
        "systems": systems,
        "evaluators": evaluators,
        "engine": engine,
        "passes": passes,
        "output_dir": output_dir
    }

def main():
    ap = argparse.ArgumentParser(description="Optimize an RT-DETR ONNX model with Olive")
    ap.add_argument("--model", required=True, help="Path to the ONNX model file.")
    ap.add_argument("--out", default="olive_out", help="Directory to save the optimized model.")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device to target.")
    ap.add_argument("--quantize", action="store_true", help="Enable INT8 static quantization.")
    ap.add_argument("--calib-dir", help="Directory with calibration images for quantization.")
    ap.add_argument("--opt-level", choices=["basic", "extended", "all"], default="all", help="ONNX Runtime optimization level.")
    args = ap.parse_args()

    calib_path = str(Path(args.calib_dir).resolve()) if args.calib_dir else None

    config = make_config(
        model_path=str(Path(args.model).resolve()),
        output_dir=str(Path(args.out).resolve()),
        device=args.device,
        do_quant=args.quantize,
        calib_dir=calib_path,
        optimization_level=args.opt_level
    )

    print("=== Olive configuration ===")
    print(json.dumps(config, indent=4))
    print("=========================\n")

    olive_run(config)

    print(f"\n✅ Done. Optimized model saved to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
