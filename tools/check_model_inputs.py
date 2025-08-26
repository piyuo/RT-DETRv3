import onnx
import onnxruntime as ort
import numpy as np

# Load and inspect your ONNX model
model_path = "output/rtdetrv3_r18vd_6x.onnx"

print("=== ONNX Model Structure ===")
model = onnx.load(model_path)

print("Model Inputs:")
for i, input in enumerate(model.graph.input):
    shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in input.type.tensor_type.shape.dim]
    print(f"  {i}: {input.name} -> shape: {shape}")

print("\nModel Outputs:")
for i, output in enumerate(model.graph.output):
    shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in output.type.tensor_type.shape.dim]
    print(f"  {i}: {output.name} -> shape: {shape}")

# Check with ORT session
print("\n=== ONNXRuntime Session Info ===")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("Session Inputs:")
for input_info in session.get_inputs():
    print(f"  {input_info.name}: {input_info.shape} ({input_info.type})")

print("Session Outputs:")
for output_info in session.get_outputs():
    print(f"  {output_info.name}: {output_info.shape} ({output_info.type})")

# Test with your working inference script format
print("\n=== Testing Input Format ===")
try:
    # Create dummy inputs matching your working script
    dummy_image = np.random.randn(1, 3, 640, 640).astype(np.float32)
    dummy_im_shape = np.array([[640, 640]], dtype=np.float32)
    dummy_scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

    # Try different input combinations
    input_names = [inp.name for inp in session.get_inputs()]

    print(f"Required input names: {input_names}")

    # Try to match inputs
    input_feed = {}
    for name in input_names:
        if name == "image" or name == "images":
            input_feed[name] = dummy_image
            print(f"  Mapped {name} -> image tensor")
        elif "shape" in name.lower():
            input_feed[name] = dummy_im_shape
            print(f"  Mapped {name} -> im_shape tensor")
        elif "scale" in name.lower():
            input_feed[name] = dummy_scale_factor
            print(f"  Mapped {name} -> scale_factor tensor")
        else:
            print(f"  Unknown input: {name}")

    # Test inference
    outputs = session.run(None, input_feed)
    print("✅ Test inference successful!")

except Exception as e:
    print(f"❌ Test inference failed: {e}")