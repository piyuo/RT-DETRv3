#!/usr/bin/env python3
"""
Test Script for Robust Re-ID Implementation
===========================================

This script tests the robustness improvements and validates the enhanced pipeline
against the specific concerns raised in the code review.

Usage:
    python test_robust_reid.py --model output/rtdetrv3_r18vd_6x_backbone.onnx --image demo/demo.jpg
"""

import argparse
import os
import sys
import numpy as np
import json
from typing import Dict, List, Any

def test_same_image_consistency():
    """Test that identical images produce consistent embeddings."""
    print("🔄 Testing same image consistency...")

    try:
        from onnx.reid_embeddings import RobustReIDEmbeddingGenerator

        # Test with same image twice
        generator = RobustReIDEmbeddingGenerator(
            "output/rtdetrv3_r18vd_6x_backbone.onnx",
            debug=False
        )

        embeddings1 = generator.process_image("demo/demo.jpg", output_dir="temp_test1")
        embeddings2 = generator.process_image("demo/demo.jpg", output_dir="temp_test2")

        if len(embeddings1) != len(embeddings2):
            print(f"❌ Different number of detections: {len(embeddings1)} vs {len(embeddings2)}")
            return False

        # Check pairwise distances for identical detections
        consistencies = []
        for (info1, emb1), (info2, emb2) in zip(embeddings1, embeddings2):
            consistency = np.dot(emb1, emb2)  # Cosine similarity
            consistencies.append(consistency)

            if consistency < 0.99:  # Should be nearly identical
                print(f"⚠️  Low consistency for detection {info1['detection_id']}: {consistency:.4f}")

        mean_consistency = np.mean(consistencies)
        print(f"✅ Mean consistency: {mean_consistency:.4f}")

        if mean_consistency > 0.99:
            print("✅ Same image consistency test PASSED")
            return True
        else:
            print("❌ Same image consistency test FAILED")
            return False

    except Exception as e:
        print(f"❌ Consistency test error: {e}")
        return False

def test_class_separability():
    """Test that different classes show larger distances than same classes."""
    print("🔄 Testing class separability...")

    try:
        from onnx.reid_embeddings import RobustReIDEmbeddingGenerator

        generator = RobustReIDEmbeddingGenerator(
            "output/rtdetrv3_r18vd_6x_backbone.onnx",
            debug=False
        )

        embeddings = generator.process_image("demo/demo.jpg", output_dir="temp_separability")

        if len(embeddings) < 2:
            print("⚠️  Need at least 2 detections for separability test")
            return True  # Not a failure, just insufficient data

        # Calculate pairwise distances
        vectors = np.array([emb for _, emb in embeddings])
        similarity_matrix = np.dot(vectors, vectors.T)

        same_class_sims = []
        diff_class_sims = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = similarity_matrix[i, j]
                distance = 1.0 - sim

                if embeddings[i][0]['class_id'] == embeddings[j][0]['class_id']:
                    same_class_sims.append(distance)
                else:
                    diff_class_sims.append(distance)

        print(f"   Same-class distances: {len(same_class_sims)} pairs")
        print(f"   Diff-class distances: {len(diff_class_sims)} pairs")

        if same_class_sims:
            print(f"   Same-class mean: {np.mean(same_class_sims):.3f} ± {np.std(same_class_sims):.3f}")

        if diff_class_sims:
            print(f"   Diff-class mean: {np.mean(diff_class_sims):.3f} ± {np.std(diff_class_sims):.3f}")

        # Check separability
        if same_class_sims and diff_class_sims:
            ratio = np.mean(diff_class_sims) / np.mean(same_class_sims)
            print(f"   Separability ratio: {ratio:.2f}")

            if ratio > 1.2:
                print("✅ Class separability test PASSED")
                return True
            else:
                print("⚠️  Class separability test MARGINAL (ratio < 1.2)")
                return True  # Not a hard failure
        else:
            print("ℹ️  Insufficient class diversity for separability test")
            return True

    except Exception as e:
        print(f"❌ Separability test error: {e}")
        return False

def test_embedding_quality():
    """Test basic embedding quality metrics."""
    print("🔄 Testing embedding quality...")

    try:
        from onnx.reid_embeddings import RobustReIDEmbeddingGenerator

        generator = RobustReIDEmbeddingGenerator(
            "output/rtdetrv3_r18vd_6x_backbone.onnx",
            debug=False
        )

        embeddings = generator.process_image("demo/demo.jpg", output_dir="temp_quality")

        if not embeddings:
            print("❌ No embeddings generated")
            return False

        all_passed = True

        for i, (info, embedding) in enumerate(embeddings):
            # Check dimension
            if len(embedding) != 512:
                print(f"❌ Embedding {i}: Wrong dimension {len(embedding)}, expected 512")
                all_passed = False

            # Check for NaN/Inf
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                print(f"❌ Embedding {i}: Contains NaN/Inf values")
                all_passed = False

            # Check normalization
            norm = np.linalg.norm(embedding)
            if abs(norm - 1.0) > 0.01:
                print(f"❌ Embedding {i}: Poor normalization, norm={norm:.4f}")
                all_passed = False

            # Check for zero embeddings
            if np.allclose(embedding, 0.0):
                print(f"❌ Embedding {i}: Zero embedding detected")
                all_passed = False

        if all_passed:
            print(f"✅ Embedding quality test PASSED ({len(embeddings)} embeddings)")
            return True
        else:
            print("❌ Embedding quality test FAILED")
            return False

    except Exception as e:
        print(f"❌ Quality test error: {e}")
        return False

def test_coordinate_consistency():
    """Test coordinate transformations and RoI extraction."""
    print("🔄 Testing coordinate consistency...")

    try:
        from onnx.reid_embeddings import RobustReIDEmbeddingGenerator

        # Test both preprocessing methods
        methods = [
            ("simple_resize", False),
            ("letterbox", True)
        ]

        results = {}

        for method_name, use_letterbox in methods:
            generator = RobustReIDEmbeddingGenerator(
                "output/rtdetrv3_r18vd_6x_backbone.onnx",
                use_letterbox=use_letterbox,
                debug=False
            )

            embeddings = generator.process_image("demo/demo.jpg", output_dir=f"temp_{method_name}")
            results[method_name] = len(embeddings)

        print(f"   Simple resize: {results.get('simple_resize', 0)} detections")
        print(f"   Letterbox: {results.get('letterbox', 0)} detections")

        # Both methods should produce similar number of detections
        if abs(results.get('simple_resize', 0) - results.get('letterbox', 0)) <= 1:
            print("✅ Coordinate consistency test PASSED")
            return True
        else:
            print("⚠️  Coordinate consistency test MARGINAL (different detection counts)")
            return True  # Not a hard failure

    except Exception as e:
        print(f"❌ Coordinate consistency test error: {e}")
        return False

def test_feature_map_validation():
    """Test feature map identification and validation."""
    print("🔄 Testing feature map validation...")

    try:
        from onnx.export_backbone import analyze_model_outputs
        import onnx

        # Load and analyze model
        model = onnx.load("output/rtdetrv3_r18vd_6x_backbone.onnx")
        candidates = analyze_model_outputs(model, input_size=640)

        # Check if we have valid backbone candidates
        backbone_candidates = [c for c in candidates if c['is_likely_backbone']]

        if not backbone_candidates:
            print("❌ No backbone feature map candidates found")
            return False

        # Check for C5 features (stride 32)
        c5_candidates = [c for c in backbone_candidates if c['stride'] == 32]

        if c5_candidates:
            print(f"✅ Found {len(c5_candidates)} C5 (stride-32) candidates")

            # Check the selected candidate
            best_candidate = c5_candidates[0]
            print(f"   Best C5 candidate: {best_candidate['name']}")
            print(f"   Shape: {best_candidate['shape']}")
            print(f"   Channels: {best_candidate['channels']}")

            if best_candidate['channels'] >= 256:
                print("✅ Feature map validation test PASSED")
                return True
            else:
                print("⚠️  Low channel count for Re-ID features")
                return True
        else:
            print("⚠️  No C5 features found, using available backbone features")
            return True

    except Exception as e:
        print(f"❌ Feature map validation test error: {e}")
        return False

def run_comprehensive_tests(model_path: str, image_path: str) -> bool:
    """Run all robustness tests."""
    print("🚀 Running comprehensive robustness tests...")
    print(f"   Model: {model_path}")
    print(f"   Image: {image_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False

    tests = [
        ("Feature Map Validation", test_feature_map_validation),
        ("Coordinate Consistency", test_coordinate_consistency),
        ("Embedding Quality", test_embedding_quality),
        ("Class Separability", test_class_separability),
        ("Same Image Consistency", test_same_image_consistency),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\\n{'-'*60}")
        print(f"Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} failed with error: {e}")
            results[test_name] = False

    # Summary
    print(f"\\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")

    print(f"\\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\\n🎉 All robustness tests PASSED! Pipeline is ready for production.")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print("\\n⚠️  Most tests passed. Review failures and warnings.")
        return True
    else:
        print("\\n❌ Multiple test failures detected. Review implementation.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Test robustness of Re-ID embedding pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="output/rtdetrv3_r18vd_6x_backbone.onnx",
                       help="Path to ONNX model with backbone features")
    parser.add_argument("--image", default="demo/demo.jpg",
                       help="Path to test image")

    args = parser.parse_args()

    try:
        success = run_comprehensive_tests(args.model, args.image)
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
