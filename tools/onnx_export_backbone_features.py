# tools/onnx_export_backbone_features.py
# Automatically identify backbone C3/C4/C5 in RT-DETRv3 and expose them as ONNX outputs

import onnx
from onnx import helper, shape_inference

onnx_path = "output/rtdetrv3_r18vd_6x_raw.onnx"
export_path = "output/rtdetrv3_r18vd_6x_backbone.onnx"

# --- Load & infer shapes ---
model = onnx.load(onnx_path)
model = shape_inference.infer_shapes(model)
graph = model.graph

input_size = 640  # assume square input (your RT-DETRv3 default)

c5_candidates = []

for vi in list(graph.value_info) + list(graph.output):
    if not vi.type.tensor_type.shape.dim:
        continue
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    if len(dims) == 4:  # [N,C,H,W]
        N, C, H, W = dims
        if H > 0 and W > 0:
            stride = input_size // H
            if stride == 32:  # only deepest feature map
                c5_candidates.append((vi.name, dims))

# --- Pick the largest-channel stride-32 (C5) ---
c5_candidates.sort(key=lambda x: x[1][1], reverse=True)  # sort by channel count
if not c5_candidates:
    raise RuntimeError("❌ No stride-32 (C5) feature map found in graph.")

c5_name, c5_shape = c5_candidates[0]
print(f"✅ Found C5: {c5_name} shape={c5_shape}")

# --- Append as graph output ---
graph.output.append(helper.ValueInfoProto(name=c5_name))

onnx.save(model, export_path)
print(f"✅ Saved model with C5 backbone output: {export_path}")
