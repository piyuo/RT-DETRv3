#!/usr/bin/env python3
"""
Re-ID Pipeline Validation Script
================================

This script validates the RT-DETRv3 Re-ID embedding pipeline against the robustness concerns
identified in the code review. It performs comprehensive checks and generates a validation report.

Validation checks:
1. Feature map selection correctness
2. Detection tensor format verification
3. Coordinate space consistency
4. Input feeding correctness
5. RoI extraction validation
6. Embedding quality assessment

Usage:
    python tools/validate_reid_pipeline.py --model output/rtdetrv3_r18vd_6x_backbone.onnx --image demo/demo.jpg --feature-map-name Concat.3
"""

import argparse
import json
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Any
import warnings

# Import the robust generator
try:
    from reid_embeddings import RobustReIDEmbeddingGenerator, COCO_CLASS_LOOKUP
    from export_backbone import analyze_model_outputs
    import onnxruntime as ort
    import onnx
    import cv2
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

class ReIDPipelineValidator:
    """Comprehensive validator for Re-ID embedding pipeline."""

    def __init__(self, model_path: str, debug: bool = True):
        self.model_path = model_path
        self.debug = debug
        self.validation_results = {
            'model_path': model_path,
            'validation_timestamp': None,
            'checks': {},
            'overall_status': 'UNKNOWN',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }

    def validate_model_structure(self) -> Dict[str, Any]:
        """Validate model structure and feature map selection."""
        print("🔍 Validating model structure...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Load and analyze model
            model = onnx.load(self.model_path)
            candidates = analyze_model_outputs(model, input_size=640)

            check_result['details']['total_candidates'] = len(candidates)
            check_result['details']['backbone_candidates'] = len([c for c in candidates if c['is_likely_backbone']])

            # Check if we have good backbone candidates
            if not any(c['is_likely_backbone'] for c in candidates):
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No likely backbone feature map candidates found")

            # Check for stride 32 (C5) availability
            stride_32_candidates = [c for c in candidates if c['stride'] == 32 and c['is_likely_backbone']]
            if stride_32_candidates:
                check_result['details']['has_c5_features'] = True
                check_result['details']['c5_candidate'] = stride_32_candidates[0]['name']
            else:
                check_result['details']['has_c5_features'] = False
                check_result['issues'].append("No stride-32 (C5) backbone features found - may impact Re-ID quality")

            # Check for potential confusion with detection outputs
            detection_like = [c for c in candidates if any(keyword in c['name'].lower()
                                                         for keyword in ['detect', 'pred', 'bbox', 'cls'])]
            if detection_like:
                check_result['details']['potential_confusion'] = [c['name'] for c in detection_like]
                check_result['issues'].append(f"Found {len(detection_like)} detection-like outputs that could be confused with backbone features")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to analyze model: {e}")

        return check_result

    def validate_detection_format(self, generator: RobustReIDEmbeddingGenerator,
                                 image_path: str) -> Dict[str, Any]:
        """Validate detection tensor format and coordinate consistency."""
        print("🔍 Validating detection format...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run a small inference to get detection format
            original_image, input_feed = generator.preprocess_image(image_path)
            detections, feature_map = generator.run_inference(input_feed)

            if not detections:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No detections returned from model")
                return check_result

            # Analyze first few detections
            sample_size = min(5, len(detections))
            sample_detections = detections[:sample_size]

            check_result['details']['sample_size'] = sample_size
            check_result['details']['detection_samples'] = []

            for i, det in enumerate(sample_detections):
                det_info = {
                    'index': i,
                    'length': len(det),
                    'values': det[:8].tolist() if len(det) >= 8 else det.tolist()
                }

                # Validate expected format [cls, conf, x1, y1, x2, y2, ...]
                if len(det) >= 6:
                    cls_id, conf, x1, y1, x2, y2 = det[:6]

                    # Check class ID range
                    if not (0 <= cls_id < 100):
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Suspicious class_id: {cls_id}")

                    # Check confidence range
                    if not (0 <= conf <= 1):
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Confidence out of range [0,1]: {conf}")

                    # Check bbox validity
                    if not (x1 < x2 and y1 < y2):
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Invalid bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                    # Check coordinate magnitude (should be in input image space)
                    if max(x1, y1, x2, y2) > 2000 or min(x1, y1, x2, y2) < -100:
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Unusual coordinate magnitude: {[x1, y1, x2, y2]}")

                    det_info['parsed'] = {
                        'class_id': cls_id,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    }

                check_result['details']['detection_samples'].append(det_info)

            # Count issues across samples
            total_issues = sum(len(det.get('issues', [])) for det in check_result['details']['detection_samples'])
            if total_issues > 0:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"Found {total_issues} detection format issues in sample")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate detection format: {e}")

        return check_result

    def validate_coordinate_consistency(self, generator: RobustReIDEmbeddingGenerator,
                                      image_path: str, feature_map_name: str = None) -> Dict[str, Any]:
        """Validate coordinate space transformations and consistency."""
        print("🔍 Validating coordinate consistency...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Process with both letterbox and simple resize to compare
            generators = {}

            # Test simple resize
            gen_simple = RobustReIDEmbeddingGenerator(
                self.model_path, use_letterbox=False, debug=False, feature_map_name=feature_map_name
            )
            generators['simple_resize'] = gen_simple

            # Test letterbox
            gen_letterbox = RobustReIDEmbeddingGenerator(
                self.model_path, use_letterbox=True, debug=False, feature_map_name=feature_map_name
            )
            generators['letterbox'] = gen_letterbox

            for method_name, gen in generators.items():
                method_result = {'preprocessing_method': method_name}

                # Get preprocessing info
                original_image, input_feed = gen.preprocess_image(image_path)
                detections, feature_map = gen.run_inference(input_feed)
                filtered_detections = gen.filter_detections(detections, conf_threshold=0.3)

                if filtered_detections:
                    scaled_detections = gen.scale_bboxes_to_feature_space(filtered_detections, feature_map.shape)

                    method_result.update({
                        'original_image_shape': original_image.shape,
                        'feature_map_shape': list(feature_map.shape),
                        'num_detections': len(filtered_detections),
                        'num_valid_scaled': len(scaled_detections),
                        'letterbox_info': gen.letterbox_info,
                        'sample_scaling': []
                    })

                    # Analyze scaling for first few detections
                    for i, (cls_id, conf, orig_bbox, scaled_bbox) in enumerate(scaled_detections[:3]):
                        scaling_info = {
                            'detection_id': i,
                            'original_bbox': orig_bbox,
                            'scaled_bbox': scaled_bbox,
                            'scaling_factors': [
                                scaled_bbox[0] / orig_bbox[0] if orig_bbox[0] != 0 else 0,
                                scaled_bbox[1] / orig_bbox[1] if orig_bbox[1] != 0 else 0
                            ]
                        }
                        method_result['sample_scaling'].append(scaling_info)

                    # Check for invalid scaled regions
                    invalid_scaled = sum(1 for _, _, _, sb in scaled_detections
                                       if sb[2] <= sb[0] or sb[3] <= sb[1])
                    if invalid_scaled > 0:
                        method_result['invalid_scaled_count'] = invalid_scaled
                        check_result['issues'].append(f"{method_name}: {invalid_scaled} invalid scaled regions")

                check_result['details'][method_name] = method_result

            # Compare methods
            if 'simple_resize' in check_result['details'] and 'letterbox' in check_result['details']:
                simple_count = check_result['details']['simple_resize'].get('num_valid_scaled', 0)
                letterbox_count = check_result['details']['letterbox'].get('num_valid_scaled', 0)

                if abs(simple_count - letterbox_count) > 1:
                    check_result['issues'].append(f"Large difference in valid detections between methods: simple={simple_count}, letterbox={letterbox_count}")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate coordinate consistency: {e}")

        return check_result

    def validate_roi_extraction(self, generator: RobustReIDEmbeddingGenerator,
                               image_path: str) -> Dict[str, Any]:
        """Validate RoI extraction quality and consistency."""
        print("🔍 Validating RoI extraction...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run full pipeline
            embeddings = generator.process_image(image_path, conf_threshold=0.3, output_dir="onnx/validation")

            if not embeddings:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No embeddings generated")
                return check_result

            check_result['details']['num_embeddings'] = len(embeddings)
            check_result['details']['roi_analysis'] = []

            very_small_rois = 0
            zero_embeddings = 0
            invalid_embeddings = 0

            for i, (detection_info, embedding) in enumerate(embeddings):
                roi_shape = detection_info['roi_shape']
                roi_analysis = {
                    'detection_id': i,
                    'class_id': detection_info['class_id'],
                    'roi_shape': list(roi_shape),
                    'roi_area': roi_shape[1] * roi_shape[2] if len(roi_shape) >= 3 else 0,
                    'embedding_stats': {
                        'min': float(embedding.min()),
                        'max': float(embedding.max()),
                        'mean': float(embedding.mean()),
                        'std': float(embedding.std()),
                        'norm': float(np.linalg.norm(embedding))
                    }
                }

                # Check for very small RoIs
                if roi_analysis['roi_area'] <= 4:  # 2x2 or smaller
                    very_small_rois += 1
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append("Very small RoI area")

                # Check for zero embeddings
                if np.allclose(embedding, 0.0):
                    zero_embeddings += 1
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append("Zero embedding")

                # Check for invalid embeddings
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    invalid_embeddings += 1
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append("Invalid embedding (NaN/Inf)")

                # Check norm (should be ~1.0 after normalization)
                expected_norm = 1.0
                norm_deviation = abs(roi_analysis['embedding_stats']['norm'] - expected_norm)
                if norm_deviation > 0.01:
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append(f"Unexpected norm: {roi_analysis['embedding_stats']['norm']:.4f}")

                check_result['details']['roi_analysis'].append(roi_analysis)

            # Summary statistics
            check_result['details']['quality_stats'] = {
                'very_small_rois': very_small_rois,
                'zero_embeddings': zero_embeddings,
                'invalid_embeddings': invalid_embeddings,
                'total_embeddings': len(embeddings)
            }

            # Quality thresholds
            if very_small_rois > len(embeddings) * 0.3:  # More than 30% very small
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"High percentage of very small RoIs: {very_small_rois}/{len(embeddings)}")

            if zero_embeddings > 0:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"Found {zero_embeddings} zero embeddings")

            if invalid_embeddings > 0:
                check_result['status'] = 'FAIL'
                check_result['issues'].append(f"Found {invalid_embeddings} invalid embeddings")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate RoI extraction: {e}")

        return check_result

    def validate_embedding_quality(self, generator: RobustReIDEmbeddingGenerator,
                                  image_path: str) -> Dict[str, Any]:
        """Validate embedding quality and separability."""
        print("🔍 Validating embedding quality...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run pipeline multiple times to check consistency
            embeddings_runs = []
            for run in range(2):  # Run twice to check consistency
                embeddings = generator.process_image(image_path, conf_threshold=0.3, output_dir=f"onnx/validation/run_{run}")
                if embeddings:
                    embeddings_runs.append(embeddings)

            if not embeddings_runs:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No embeddings generated in any run")
                return check_result

            # Analyze primary run
            primary_embeddings = embeddings_runs[0]
            vectors = np.array([emb for _, emb in primary_embeddings])

            check_result['details']['num_embeddings'] = len(primary_embeddings)
            check_result['details']['embedding_dimension'] = len(primary_embeddings[0][1]) if primary_embeddings else 0

            # Analyze embedding statistics
            norms = [np.linalg.norm(emb) for _, emb in primary_embeddings]
            check_result['details']['norm_stats'] = {
                'mean': float(np.mean(norms)),
                'std': float(np.std(norms)),
                'min': float(np.min(norms)),
                'max': float(np.max(norms))
            }

            # Check norm consistency (should all be ~1.0)
            norm_deviations = [abs(norm - 1.0) for norm in norms]
            max_norm_deviation = max(norm_deviations) if norm_deviations else 0
            if max_norm_deviation > 0.01:
                check_result['issues'].append(f"Large norm deviations detected: max={max_norm_deviation:.4f}")

            # Analyze class separability if multiple classes present
            if len(primary_embeddings) >= 2:
                similarity_matrix = np.dot(vectors, vectors.T)

                same_class_similarities = []
                diff_class_similarities = []

                for i in range(len(primary_embeddings)):
                    for j in range(i + 1, len(primary_embeddings)):
                        sim = similarity_matrix[i, j]
                        if primary_embeddings[i][0]['class_id'] == primary_embeddings[j][0]['class_id']:
                            same_class_similarities.append(sim)
                        else:
                            diff_class_similarities.append(sim)

                separability_analysis = {
                    'same_class_count': len(same_class_similarities),
                    'diff_class_count': len(diff_class_similarities)
                }

                if same_class_similarities:
                    separability_analysis['same_class_stats'] = {
                        'mean': float(np.mean(same_class_similarities)),
                        'std': float(np.std(same_class_similarities))
                    }

                if diff_class_similarities:
                    separability_analysis['diff_class_stats'] = {
                        'mean': float(np.mean(diff_class_similarities)),
                        'std': float(np.std(diff_class_similarities))
                    }

                # Calculate separability ratio
                if same_class_similarities and diff_class_similarities:
                    mean_diff = np.mean(diff_class_similarities)
                    mean_same = np.mean(same_class_similarities)
                    separability_ratio = mean_diff / mean_same if mean_same > 0 else float('inf')

                    separability_analysis['separability_ratio'] = float(separability_ratio)

                    if separability_ratio < 1.2:
                        check_result['status'] = 'WARN'
                        check_result['issues'].append(f"Poor class separability: ratio={separability_ratio:.2f} (should be > 1.2)")

                check_result['details']['separability_analysis'] = separability_analysis

            # Check consistency between runs
            if len(embeddings_runs) >= 2:
                # Compare embeddings for identical detections
                run1_embeddings = embeddings_runs[0]
                run2_embeddings = embeddings_runs[1]

                if len(run1_embeddings) == len(run2_embeddings):
                    consistency_scores = []
                    for (_, emb1), (_, emb2) in zip(run1_embeddings, run2_embeddings):
                        consistency = np.dot(emb1, emb2)  # Cosine similarity
                        consistency_scores.append(consistency)

                    mean_consistency = np.mean(consistency_scores)
                    check_result['details']['consistency_analysis'] = {
                        'mean_consistency': float(mean_consistency),
                        'consistency_scores': [float(s) for s in consistency_scores]
                    }

                    if mean_consistency < 0.95:  # Should be very similar
                        check_result['status'] = 'WARN'
                        check_result['issues'].append(f"Low consistency between runs: {mean_consistency:.3f}")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate embedding quality: {e}")

        return check_result

    def run_comprehensive_validation(self, image_path: str, feature_map_name: str = None) -> Dict[str, Any]:
        """Run all validation checks and generate comprehensive report."""
        print("🔄 Starting comprehensive Re-ID pipeline validation...")
        print(f"   Model: {self.model_path}")
        print(f"   Test image: {image_path}")
        if feature_map_name:
            print(f"   Feature map: {feature_map_name}")

        # Initialize generator for testing
        generator = RobustReIDEmbeddingGenerator(self.model_path, debug=False, feature_map_name=feature_map_name)

        # Run all validation checks
        checks = {}

        checks['model_structure'] = self.validate_model_structure()
        checks['detection_format'] = self.validate_detection_format(generator, image_path)
        checks['coordinate_consistency'] = self.validate_coordinate_consistency(generator, image_path, feature_map_name)
        checks['roi_extraction'] = self.validate_roi_extraction(generator, image_path)
        checks['embedding_quality'] = self.validate_embedding_quality(generator, image_path)

        # Determine overall status
        statuses = [check['status'] for check in checks.values()]
        if 'ERROR' in statuses or 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARN' in statuses:
            overall_status = 'WARN'
        else:
            overall_status = 'PASS'

        # Collect all issues
        critical_issues = []
        warnings = []

        for check_name, check_result in checks.items():
            for issue in check_result.get('issues', []):
                if check_result['status'] in ['ERROR', 'FAIL']:
                    critical_issues.append(f"{check_name}: {issue}")
                else:
                    warnings.append(f"{check_name}: {issue}")

        # Generate recommendations
        recommendations = self._generate_recommendations(checks)

        self.validation_results.update({
            'validation_timestamp': np.datetime64('now').astype(str),
            'checks': checks,
            'overall_status': overall_status,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'recommendations': recommendations
        })

        return self.validation_results

    def _generate_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Model structure recommendations
        if checks['model_structure']['status'] != 'PASS':
            if not checks['model_structure']['details'].get('has_c5_features', False):
                recommendations.append("Consider using a model with C5 (stride-32) backbone features for optimal Re-ID performance")

            if checks['model_structure']['details'].get('potential_confusion'):
                recommendations.append("Use --feature-map-name to explicitly specify backbone feature to avoid confusion with detection outputs")

        # Detection format recommendations
        if checks['detection_format']['status'] != 'PASS':
            recommendations.append("Verify detection tensor format - consider using --detection-layout parameter")
            recommendations.append("Inspect raw detection outputs and confirm coordinate format matches expectations")

        # Coordinate consistency recommendations
        if checks['coordinate_consistency']['status'] != 'PASS':
            recommendations.append("Consider using letterbox preprocessing (--use-letterbox) if model was trained with aspect ratio preservation")
            recommendations.append("Verify that coordinate scaling matches model training preprocessing")

        # RoI extraction recommendations
        if checks['roi_extraction']['status'] != 'PASS':
            roi_stats = checks['roi_extraction']['details'].get('quality_stats', {})
            if roi_stats.get('very_small_rois', 0) > 0:
                recommendations.append("Consider using higher resolution feature maps (C4 instead of C5) for small object Re-ID")
            if roi_stats.get('zero_embeddings', 0) > 0:
                recommendations.append("Check feature map activation patterns - zero embeddings may indicate inactive regions")

        # Embedding quality recommendations
        if checks['embedding_quality']['status'] != 'PASS':
            separability = checks['embedding_quality']['details'].get('separability_analysis', {})
            if separability.get('separability_ratio', float('inf')) < 1.2:
                recommendations.append("Poor class separability detected - consider fine-tuning model with Re-ID supervision")
                recommendations.append("Try combining multiple feature map levels (C3+C4+C5) for better discriminative power")

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Validation passed - pipeline is ready for production use")
            recommendations.append("Consider running validation on multiple diverse images to ensure robustness")
        else:
            recommendations.append("Run validation again after implementing recommended fixes")

        return recommendations

    def save_report(self, output_path: str):
        """Save validation report to JSON file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        json_compatible = self._convert_numpy_types(self.validation_results)

        with open(output_path, 'w') as f:
            json.dump(json_compatible, f, indent=2)
        print(f"📋 Validation report saved: {output_path}")

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def print_summary(self):
        """Print a human-readable validation summary."""
        print(f"{'='*80}")
        print(f"RE-ID PIPELINE VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Overall Status: {self.validation_results['overall_status']}")
        print(f"Model: {self.validation_results['model_path']}")
        print(f"Timestamp: {self.validation_results['validation_timestamp']}")

        print(f"📊 Check Results:")
        for check_name, check_result in self.validation_results['checks'].items():
            status_emoji = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥'}
            emoji = status_emoji.get(check_result['status'], '❓')
            print(f"   {emoji} {check_name.replace('_', ' ').title()}: {check_result['status']}")

        if self.validation_results['critical_issues']:
            print(f"🚨 Critical Issues:")
            for issue in self.validation_results['critical_issues']:
                print(f"   - {issue}")

        if self.validation_results['warnings']:
            print(f"⚠️  Warnings:")
            for warning in self.validation_results['warnings']:
                print(f"   - {warning}")

        print(f"💡 Recommendations:")
        for rec in self.validation_results['recommendations']:
            print(f"   - {rec}")

        print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(
        description="Validate Re-ID embedding pipeline robustness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="output/rtdetrv3_r18vd_6x_backbone.onnx",
                       help="Path to ONNX model with backbone features")
    parser.add_argument("--image", default="demo/demo.jpg",
                       help="Path to test image")
    parser.add_argument("--output", default="output/reid_validation_report.json",
                       help="Path to save validation report")
    parser.add_argument("--feature-map-name",
                       help="Explicitly specify backbone feature map name (e.g., Concat.5 for C3, Concat.3 for C4)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return 1

    if not os.path.exists(args.image):
        print(f"❌ Test image not found: {args.image}")
        return 1

    try:
        # Run validation
        validator = ReIDPipelineValidator(args.model, debug=args.debug)
        results = validator.run_comprehensive_validation(args.image, args.feature_map_name)

        # Save report
        validator.save_report(args.output)

        # Print summary
        validator.print_summary()

        # Return appropriate exit code
        if results['overall_status'] == 'FAIL':
            print("❌ Validation failed - critical issues detected")
            return 1
        elif results['overall_status'] == 'WARN':
            print("⚠️  Validation completed with warnings")
            return 0
        else:
            print("✅ Validation passed successfully")
            return 0

    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
