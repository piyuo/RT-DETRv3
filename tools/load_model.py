# tools/load_model.py
import onnx

onnx_path = "output_inference/rtdetrv3_r18vd_6x.onnx"

print(f"Loading: {onnx_path}")
model = onnx.load(onnx_path)

# Optional: list concats to eyeball inputs after patch
for n in model.graph.node:
    if n.op_type == "Concat":
        print(f"Checking {n.name} with inputs {list(n.input)}")

print("Running ONNX checker...")
onnx.checker.check_model(model)
print("Model is valid ✅")

print(f"\nInputs: {[i.name for i in model.graph.input]}")
print(f"Outputs: {[o.name for o in model.graph.output]}")
print(f"Nodes: {len(model.graph.node)}")
