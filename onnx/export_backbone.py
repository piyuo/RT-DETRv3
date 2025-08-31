# onnx/export_backbone.py

#!/usr/bin/env python3
"""
Enhanced ONNX Backbone Features Exporter for RT-DETRv3
=======================================================

This script provides robust detection and export of backbone feature maps (C) from RT-DETRv3 ONNX models.
It includes enhanced validation, multiple candidate analysis, and explicit naming for Re-ID embedding generation.

Key improvements:
- Validates that the exported feature matches expected backbone characteristics
- Provides detailed analysis of all potential feature map candidates
- Supports explicit stride verification
- Adds metadata to track the export process

Usage:
    python onnx/export_backbone.py --input onnx/input/rtdetrv3_r18vd_6x.onnx --output onnx/backbone/rtdetrv3_r18vd_6x.onnx
"""

import onnx
from onnx import helper, shape_inference
import argparse
import sys
from typing import List, Tuple, Dict

def analyze_model_outputs(model: onnx.ModelProto, input_size: int = 640) -> List[Dict]:
    """Analyze all model outputs to identify potential backbone feature maps.

    Args:
        model: ONNX model to analyze
        input_size: Expected input image size

    Returns:
        List of candidate feature map information
    """
    graph = model.graph
    candidates = []

    print("🔍 Analyzing model outputs for backbone feature candidates...")
    print(f"   Input size assumption: {input_size}x{input_size}")

    # Collect all value infos (intermediate and outputs)
    all_value_infos = list(graph.value_info) + list(graph.output)

    for vi in all_value_infos:
        if not vi.type.tensor_type.shape.dim:
            continue

        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]

        # Look for 4D tensors (potential feature maps)
        if len(dims) == 4:
            N, C, H, W = dims

            # Skip if dimensions are not specified
            if H <= 0 or W <= 0 or C <= 0:
                continue

            # Calculate stride
            stride = input_size // H if H > 0 else float('inf')

            # Analyze characteristics
            is_likely_backbone = (
                10 <= H <= 80 and  # Reasonable spatial size
                10 <= W <= 80 and
                C >= 64 and        # Reasonable channel count for backbone
                stride in [8, 16, 32, 64]  # Common backbone strides
            )

            candidate_info = {
                'name': vi.name,
                'shape': dims,
                'channels': C,
                'spatial_size': (H, W),
                'stride': stride,
                'is_likely_backbone': is_likely_backbone,
                'area': H * W,
                'channel_to_spatial_ratio': C / (H * W) if H * W > 0 else float('inf')
            }

            candidates.append(candidate_info)

    # Sort candidates by likelihood (stride preference: 16 > 32 > 8, then by channels)
    # C4 features (stride 16) are preferred for ReID embeddings due to better spatial resolution
    def candidate_score(candidate):
        stride_score = {16: 1000, 32: 900, 8: 800}.get(candidate['stride'], 0)
        channel_score = candidate['channels']
        return stride_score + channel_score

    candidates.sort(key=candidate_score, reverse=True)

    print(f"📊 Found {len(candidates)} potential feature map candidates:")
    print("   Name                          | Shape           | Stride | Channels | Backbone?")
    print("   " + "-" * 80)

    for i, candidate in enumerate(candidates):
        marker = "✅" if candidate['is_likely_backbone'] else "❌"
        selected = "🎯 " if i == 0 and candidate['is_likely_backbone'] else "   "

        print(f"{selected}{marker} {candidate['name']:<25} | "
              f"{str(candidate['shape']):<15} | "
              f"{candidate['stride']:<6.0f} | "
              f"{candidate['channels']:<8} | "
              f"{candidate['is_likely_backbone']}")

    return candidates

def validate_feature_map_selection(candidate: Dict, input_size: int = 640) -> Tuple[bool, List[str]]:
    """Validate that a selected feature map is appropriate for Re-ID embeddings.

    Args:
        candidate: Candidate feature map information
        input_size: Expected input image size

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    is_valid = True

    # Check stride
    expected_strides = [8, 16, 32]
    if candidate['stride'] not in expected_strides:
        warnings.append(f"Unusual stride {candidate['stride']}, expected one of {expected_strides}")
        if candidate['stride'] > 64:
            is_valid = False

    # Check channel count
    if candidate['channels'] < 64:
        warnings.append(f"Low channel count {candidate['channels']}, may produce poor embeddings")
    elif candidate['channels'] < 128:
        warnings.append(f"Moderate channel count {candidate['channels']}, consider using deeper layer")

    # Check spatial resolution
    spatial_h, spatial_w = candidate['spatial_size']
    if spatial_h < 10 or spatial_w < 10:
        warnings.append(f"Very small spatial size {spatial_h}x{spatial_w}, may limit RoI extraction quality")
        is_valid = False
    elif spatial_h < 20 or spatial_w < 20:
        warnings.append(f"Small spatial size {spatial_h}x{spatial_w}, consider using higher resolution layer")

    # Check if this looks like a detection output rather than backbone
    if 'detect' in candidate['name'].lower() or 'pred' in candidate['name'].lower():
        warnings.append(f"Name '{candidate['name']}' suggests detection output, not backbone feature")

    # Prefer C4 features (stride 16) for Re-ID
    if candidate['stride'] == 16:
        pass  # Ideal for Re-ID - good balance of spatial resolution and semantic content
    elif candidate['stride'] == 32:
        warnings.append("Using C5 features (stride 32) - good semantic content but C4 (stride 16) typically preferred for Re-ID")
    elif candidate['stride'] == 8:
        warnings.append("Using C3 features (stride 8) - may be too high resolution for efficient Re-ID")

    return is_valid, warnings

def export_backbone_features(input_path: str, output_path: str,
                           feature_map_name: str = None,
                           input_size: int = 640,
                           validate_export: bool = True) -> bool:
    """Export ONNX model with backbone feature map as additional output.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save modified model
        feature_map_name: Explicit feature map name (if None, auto-detect)
        input_size: Expected input image size
        validate_export: Whether to validate the export

    Returns:
        True if export was successful
    """
    print(f"🔄 Loading ONNX model: {input_path}")

    try:
        model = onnx.load(input_path)
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    graph = model.graph

    # Analyze candidates
    candidates = analyze_model_outputs(model, input_size)

    if not candidates:
        print("❌ No potential backbone feature map candidates found!")
        return False

    # Select feature map
    if feature_map_name:
        # Use explicit name
        selected_candidate = None
        for candidate in candidates:
            if candidate['name'] == feature_map_name:
                selected_candidate = candidate
                break

        if selected_candidate is None:
            print(f"❌ Explicit feature map '{feature_map_name}' not found in candidates!")
            available_names = [c['name'] for c in candidates]
            print(f"   Available candidates: {available_names}")
            return False
    else:
        # Auto-select best candidate
        backbone_candidates = [c for c in candidates if c['is_likely_backbone']]

        if not backbone_candidates:
            print("❌ No likely backbone candidates found!")
            print("   Consider using --feature-map-name to specify explicitly")
            return False

        selected_candidate = backbone_candidates[0]
        print(f"🎯 Auto-selected feature map: {selected_candidate['name']}")

    # Validate selection
    is_valid, warnings = validate_feature_map_selection(selected_candidate, input_size)

    print(f"🔍 Validating selected feature map: {selected_candidate['name']}")
    print(f"   Shape: {selected_candidate['shape']}")
    print(f"   Stride: {selected_candidate['stride']}")
    print(f"   Channels: {selected_candidate['channels']}")

    if warnings:
        print("⚠️  Validation warnings:")
        for warning in warnings:
            print(f"   - {warning}")

    if not is_valid:
        print("❌ Selected feature map failed validation!")
        return False

    if warnings and not feature_map_name:
        print("❓ Warnings detected. Consider using a different feature map or --feature-map-name to specify explicitly.")

        # Show alternatives
        other_candidates = [c for c in candidates if c['name'] != selected_candidate['name'] and c['is_likely_backbone']]
        if other_candidates:
            print("   Alternative candidates:")
            for alt in other_candidates[:3]:  # Show top 3 alternatives
                print(f"   - {alt['name']}: stride={alt['stride']}, channels={alt['channels']}")

    # Check if feature map is already an output
    existing_output_names = [out.name for out in graph.output]
    if selected_candidate['name'] in existing_output_names:
        print(f"✅ Feature map '{selected_candidate['name']}' is already a model output")
        print(f"   Model already suitable for Re-ID embedding generation")
        if input_path != output_path:
            print(f"💾 Copying model to output path: {output_path}")
            onnx.save(model, output_path)
        return True

    # Add feature map as output
    print(f"🔧 Adding '{selected_candidate['name']}' as model output...")

    try:
        # Create value info for the new output
        feature_value_info = helper.ValueInfoProto()
        feature_value_info.name = selected_candidate['name']

        # Add to outputs
        graph.output.append(feature_value_info)

        # Save modified model
        onnx.save(model, output_path)
        print(f"✅ Successfully exported model with backbone features: {output_path}")

    except Exception as e:
        print(f"❌ Failed to export model: {e}")
        return False

    # Validate export if requested
    if validate_export:
        print("🔍 Validating exported model...")
        try:
            exported_model = onnx.load(output_path)
            exported_outputs = [out.name for out in exported_model.graph.output]

            if selected_candidate['name'] in exported_outputs:
                print(f"✅ Export validation successful - '{selected_candidate['name']}' found in outputs")
                print(f"   Total model outputs: {len(exported_outputs)}")
            else:
                print(f"❌ Export validation failed - '{selected_candidate['name']}' not in outputs")
                return False

        except Exception as e:
            print(f"❌ Export validation failed: {e}")
            return False

    print(f"🎉 Backbone feature export completed successfully!")
    print(f"   Input model: {input_path}")
    print(f"   Output model: {output_path}")
    print(f"   Exported feature: {selected_candidate['name']}")
    print(f"   Feature shape: {selected_candidate['shape']}")
    print(f"   Recommended for Re-ID: {'Yes' if selected_candidate['stride'] == 16 else 'Consider C4 layer if available'}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced ONNX backbone features exporter for RT-DETRv3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", default="output/rtdetrv3_r18vd_6x_raw.onnx",
                       help="Path to input ONNX model")
    parser.add_argument("--output", default="output/rtdetrv3_r18vd_6x_backbone.onnx",
                       help="Path to save modified model")
    parser.add_argument("--feature-map-name", default=None,
                       help="Explicit feature map name to export (if not provided, auto-detect)")
    parser.add_argument("--input-size", type=int, default=640,
                       help="Expected input image size for stride calculation")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip export validation")
    parser.add_argument("--list-only", action="store_true",
                       help="Only list candidate feature maps, don't export")

    args = parser.parse_args()

    # Check if input file exists
    if not args.list_only and not args.input:
        print("❌ Input model path is required")
        return 1

    try:
        if args.list_only:
            # Just analyze and list candidates
            print(f"🔍 Analyzing model outputs: {args.input}")
            model = onnx.load(args.input)
            model = shape_inference.infer_shapes(model)
            candidates = analyze_model_outputs(model, args.input_size)

            if candidates:
                print(f"📋 Use --feature-map-name with one of these candidates:")
                for candidate in candidates:
                    if candidate['is_likely_backbone']:
                        print(f"   {candidate['name']} (stride={candidate['stride']}, channels={candidate['channels']})")
            else:
                print("❌ No suitable candidates found")
                return 1
        else:
            # Perform export
            success = export_backbone_features(
                input_path=args.input,
                output_path=args.output,
                feature_map_name=args.feature_map_name,
                input_size=args.input_size,
                validate_export=not args.no_validate
            )

            if not success:
                print("❌ Export failed")
                return 1

            print("📝 Next steps:")
            print("   1. Use the exported model with reid_embeddings.py")
            if args.feature_map_name:
                print(f"   2. Pass --feature-map-name '{args.feature_map_name}' for explicit validation")
            print("   3. Verify Re-ID embedding quality with your test images")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
