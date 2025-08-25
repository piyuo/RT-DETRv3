#!/usr/bin/env python3
import argparse
from pathlib import Path
from olive.workflows import run as olive_run
import json

def make_config(model_path, output_dir, device, optimization_level="all"):
    systems = {"local_system": {"type": "LocalSystem"}}

    # Minimal engine: no evaluator, uses sampler-based search strategy
    engine = {
        "host": "local_system",
        "target": "local_system",
        "cache_dir": ".cache",
        "search_strategy": {
            "execution_order": "joint",
            "sampler": "random"  # if this errors, change to: "search_algorithm": "random"
        }
        # no "evaluator" -> Olive will skip evaluations
    }

    passes = {
        "ort_opt": {
            "type": "OrtOptimization",
            # Do NOT put "engine" here. Passes inherit from the top-level engine.
            "config": {
                "optimization_level": optimization_level,  # basic | extended | all
                "target_device": device                    # cpu | gpu
            }
        }
    }

    input_model = {"type": "OnnxModel", "config": {"model_path": model_path}}

    return {
        "input_model": input_model,
        "systems": systems,
        "engine": engine,          # <- singular
        "passes": passes,
        "output_dir": output_dir
    }

def main():
    ap = argparse.ArgumentParser(description="Optimize an RT-DETR ONNX model with Olive")
    ap.add_argument("--model", required=True, help="Path to the ONNX model file.")
    ap.add_argument("--out", default="olive_out", help="Directory to save the optimized model.")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device to target.")
    ap.add_argument("--opt-level", choices=["basic", "extended", "all"], default="all",
                    help="ONNX Runtime optimization level.")
    args = ap.parse_args()

    config = make_config(
        model_path=str(Path(args.model).resolve()),
        output_dir=str(Path(args.out).resolve()),
        device=args.device,
        optimization_level=args.opt_level
    )

    print("=== Olive configuration ===")
    print(json.dumps(config, indent=4))
    print("=========================\n")

    olive_run(config)

    print(f"\n✅ Done. Optimized model saved to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
