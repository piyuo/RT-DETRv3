# tools/fix_model.py
import onnx
from onnx import helper, TensorProto, shape_inference

SRC = "output_inference/rtdetrv3_r18vd_6x.onnx"
DST = "output_inference/rtdetrv3_r18vd_6x_fixed.onnx"

def collect_shapes(model):
    shapes = {}
    def _grab(v):
        t = v.type.tensor_type
        if t.HasField("shape"):
            dims = []
            for d in t.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                else:
                    dims.append(None)
            shapes[v.name] = dims

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        _grab(vi)
    return shapes

model = onnx.load(SRC)

# First pass shape inference (best-effort)
model = shape_inference.infer_shapes(model)
shapes = collect_shapes(model)

new_nodes = []
new_inits = []

for node in model.graph.node:
    if node.op_type != "Concat":
        # keep nodes as-is
        new_nodes.append(node)
        continue

    # Determine input ranks
    in_shapes = [shapes.get(x) for x in node.input]
    ranks = [len(s) if s is not None else None for s in in_shapes]
    if None in ranks or len(set(ranks)) == 1:
        # unknown or already equal -> leave it
        new_nodes.append(node)
        continue

    max_rank = max(r for r in ranks if r is not None)
    print(f"Fixing {node.name}: ranks={ranks}")

    fixed_inputs = []
    for i, inp in enumerate(node.input):
        r = ranks[i]
        out_name = inp
        while r is not None and r < max_rank:
            axes_name = f"{node.name}_axes_{i}_{r}"
            axes_tensor = helper.make_tensor(
                name=axes_name,
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[0],  # prepend a dimension at axis 0
            )
            new_inits.append(axes_tensor)

            unsq_name = f"{inp}_unsq_rank{r}_to_{r+1}"
            unsq_node = helper.make_node(
                "Unsqueeze",
                inputs=[out_name, axes_name],
                outputs=[unsq_name],
                name=f"{node.name}_fix_unsq_{i}_{r}",
            )
            # IMPORTANT: insert Unsqueeze BEFORE the Concat (topo order)
            new_nodes.append(unsq_node)

            out_name = unsq_name
            r += 1
        fixed_inputs.append(out_name)

    # Rewrite the Concat to use the patched inputs
    new_concat = helper.make_node(
        "Concat",
        inputs=fixed_inputs,
        outputs=list(node.output),
        name=node.name,
    )
    # preserve original attributes (e.g., axis)
    for attr in node.attribute:
        new_concat.attribute.extend([onnx.helper.make_attribute(attr.name, onnx.helper.get_attribute_value(attr))])

    new_nodes.append(new_concat)

# Replace graph nodes with the rebuilt, ordered list
model.graph.ClearField("node")
model.graph.node.extend(new_nodes)

# Add any new initializers
model.graph.initializer.extend(new_inits)

# Final shape inference & save
model = shape_inference.infer_shapes(model)



# add this inside fix_model.py, after Concat fixing but before saving
from onnx import numpy_helper
import numpy as np

def fix_reshape_shapes(model):
    new_nodes = []
    new_inits = list(model.graph.initializer)

    for node in model.graph.node:
        if node.op_type != "Reshape":
            new_nodes.append(node)
            continue

        if len(node.input) != 2:
            new_nodes.append(node)
            continue

        data_inp, shape_inp = node.input
        fixed_shape_inp = shape_inp

        # check if shape input is initializer
        init = next((i for i in model.graph.initializer if i.name == shape_inp), None)
        if init:
            arr = numpy_helper.to_array(init)
            if arr.ndim != 1:
                print(f"Fixing Reshape {node.name}: shape initializer {arr.shape} -> 1D")
                flat = arr.flatten().astype(np.int64)
                new_init = numpy_helper.from_array(flat, name=shape_inp)
                # replace old initializer
                new_inits = [i for i in new_inits if i.name != shape_inp]
                new_inits.append(new_init)
        else:
            # not initializer, insert a Flatten/Reshape so it's 1D
            fixed_name = f"{shape_inp}_fixed1d"
            flatten_node = helper.make_node(
                "Reshape",
                inputs=[shape_inp, helper.make_tensor(
                    name=f"{shape_inp}_force1d_shape",
                    data_type=TensorProto.INT64,
                    dims=[1],
                    vals=[-1],
                ).name],
                outputs=[fixed_name],
                name=f"{node.name}_fix_shape_input"
            )
            # Add that constant to initializers
            const_shape = helper.make_tensor(
                name=f"{shape_inp}_force1d_shape",
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[-1],
            )
            new_inits.append(const_shape)
            new_nodes.append(flatten_node)
            fixed_shape_inp = fixed_name

        # rebuild reshape node
        new_nodes.append(helper.make_node(
            "Reshape",
            inputs=[data_inp, fixed_shape_inp],
            outputs=list(node.output),
            name=node.name
        ))

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_inits)
    return model

# call it before saving:
model = fix_reshape_shapes(model)


onnx.save(model, DST)
print("Saved generalized fixed model:", DST)
