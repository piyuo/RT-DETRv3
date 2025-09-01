#!/usr/bin/env python3
import sys
sys.path.append('tools')

import onnxruntime as ort
from onnx.rtdetr import CocoDetection, dataset_post_process
import numpy as np

def test_dataset_with_model():
    print("=== Testing Dataset with ONNX Model ===")

    # Load model
    model_path = "output/rtdetrv3_r18vd_6x.onnx"
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Load dataset
    try:
        dataset = CocoDetection(
            data_dir="dataset/coco/val2017",
            annotation_file="dataset/coco/annotations/instances_val2017.json"
        )
        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

    # Test first few samples
    for i in range(min(3, len(dataset))):
        try:
            print(f"\n--- Testing sample {i} ---")
            inputs, targets = dataset[i]

            print("Input keys and shapes:")
            for key, value in inputs.items():
                print(f"  {key}: {value.shape} ({value.dtype})")

            # Run inference
            outputs = session.run(None, inputs)
            print(f"Inference successful! Output shapes:")
            output_names = [out.name for out in session.get_outputs()]
            for name, output in zip(output_names, outputs):
                print(f"  {name}: {output.shape}")

            # Test post-processing
            output_dict = dict(zip(output_names, outputs))
            processed = dataset_post_process(output_dict)
            print(f"Post-processing successful:")
            for key, value in processed.items():
                print(f"  {key}: {value.shape if hasattr(value, 'shape') else len(value)}")

        except Exception as e:
            print(f"❌ Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✅ All tests passed! Dataset is compatible with the model.")
    return True

if __name__ == "__main__":
    success = test_dataset_with_model()
    sys.exit(0 if success else 1)