#!/usr/bin/env python3
"""
RE-ID Embeddings Generator for RT-DETRv3 with Backbone Features
==============================================================

This script processes the RT-DETR model with backbone features to generate Re-ID embeddings
for use in BoT-SORT tracking. It follows these steps:
1. Run inference to get detections and feature maps
2. Scale bounding box coordinates to feature map space
3. Crop regions of interest (RoI) from feature maps
4. Apply Global Average Pooling (GAP)
5. L2 normalize embeddings

Usage:
    python reid_embeddings.py --model output/rtdetrv3_r18vd_6x_backbone.onnx --image demo/demo.jpg
"""

import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
import json
from typing import List, Tuple, Dict, Optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available, visualization features disabled")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# COCO class names for reference
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

class ReIDEmbeddingGenerator:
    def __init__(self, model_path: str, debug: bool = True):
        """Initialize the Re-ID embedding generator.

        Args:
            model_path: Path to the ONNX model with backbone features
            debug: Whether to print debug information
        """
        self.model_path = model_path
        self.debug = debug
        self.session = None
        self.feature_map_name = None
        self.feature_map_shape = None
        self.input_size = 640  # RT-DETR default input size

        self._load_model()
        self._identify_feature_output()

    def _load_model(self):
        """Load the ONNX model and create inference session."""
        if self.debug:
            print(f"🔄 Loading ONNX model: {self.model_path}")

        sess_opts = ort.SessionOptions()
        if self.debug:
            sess_opts.log_severity_level = 0

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"]
        )

        if self.debug:
            print("✅ Model loaded successfully")
            print("\n=== MODEL INPUTS ===")
            for inp in self.session.get_inputs():
                print(f"  {inp.name}: shape={inp.shape} type={inp.type}")

            print("\n=== MODEL OUTPUTS ===")
            for out in self.session.get_outputs():
                print(f"  {out.name}: shape={out.shape} type={out.type}")

    def _identify_feature_output(self):
        """Identify which output contains the backbone feature map."""
        outputs = self.session.get_outputs()

        # Look for the backbone feature output (should be 4D tensor with spatial dimensions)
        for out in outputs:
            if len(out.shape) == 4:  # [batch, channels, height, width]
                # Check if this looks like a feature map (reasonable spatial size)
                if out.shape[2] and out.shape[3]:  # height and width are specified
                    h, w = out.shape[2], out.shape[3]
                    if 10 <= h <= 50 and 10 <= w <= 50:  # reasonable feature map size
                        self.feature_map_name = out.name
                        self.feature_map_shape = out.shape
                        break

        if self.feature_map_name is None:
            raise RuntimeError("❌ Could not identify backbone feature map output!")

        if self.debug:
            print(f"\n✅ Identified feature map output: {self.feature_map_name}")
            print(f"   Shape: {self.feature_map_shape}")

            # Calculate stride
            if self.feature_map_shape[2] > 0:
                stride = self.input_size // self.feature_map_shape[2]
                print(f"   Estimated stride: {stride}")

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Preprocess image for model inference.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (original_image, input_feed_dict)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"❌ Image not found: {image_path}")

        original_shape = img.shape
        if self.debug:
            print(f"\n=== PREPROCESSING ===")
            print(f"Original image shape: {original_shape}")

        # Prepare input tensors
        im_shape = np.array([[img.shape[0], img.shape[1]]], dtype=np.float32)
        scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

        # Resize and normalize image
        resized = cv2.resize(img, (self.input_size, self.input_size))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_tensor = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        if self.debug:
            print(f"Resized image shape: {image_tensor.shape}")
            print(f"Image tensor range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
            print(f"Image tensor mean: {image_tensor.mean():.4f}")

        # Create input feed dictionary
        input_feed = {}
        for input_node in self.session.get_inputs():
            if len(input_node.shape) == 4 and input_node.shape[1] == 3:  # image
                input_feed[input_node.name] = image_tensor
            elif "shape" in input_node.name.lower():
                input_feed[input_node.name] = im_shape
            elif "scale" in input_node.name.lower():
                input_feed[input_node.name] = scale_factor
            else:
                # Fallback: assume first 2D input is im_shape, second is scale_factor
                if len([k for k in input_feed.keys() if k != input_node.name]) == 1:
                    input_feed[input_node.name] = scale_factor
                else:
                    input_feed[input_node.name] = im_shape

        return img, input_feed

    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Tuple[List, np.ndarray]:
        """Run inference and extract detections and feature map.

        Args:
            input_feed: Dictionary of input tensors

        Returns:
            Tuple of (detections, feature_map)
        """
        if self.debug:
            print(f"\n=== INFERENCE ===")
            print("🔄 Running inference...")

        outputs = self.session.run(None, input_feed)
        output_names = [out.name for out in self.session.get_outputs()]

        if self.debug:
            print("✅ Inference completed")
            print(f"Total outputs: {len(outputs)}")

        # Extract feature map
        feature_map_idx = output_names.index(self.feature_map_name)
        feature_map = outputs[feature_map_idx]

        if self.debug:
            print(f"\n=== FEATURE MAP ===")
            print(f"Feature map shape: {feature_map.shape}")
            print(f"Feature map range: [{feature_map.min():.4f}, {feature_map.max():.4f}]")
            print(f"Feature map mean: {feature_map.mean():.4f}, std: {feature_map.std():.4f}")

        # Extract detections from other outputs
        detections = []
        for i, (output, output_name) in enumerate(zip(outputs, output_names)):
            if output_name == self.feature_map_name:
                continue

            arr = np.array(output)
            if self.debug:
                print(f"\nOutput[{i}] {output_name}: shape={arr.shape} dtype={arr.dtype}")

            # Look for detection output (should have 6 columns: class, conf, x1, y1, x2, y2)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                detections.extend(arr)
            elif arr.ndim == 3 and arr.shape[2] >= 6:
                detections.extend(arr.reshape(-1, arr.shape[-1]))

        return detections, feature_map

    def filter_detections(self, detections: List, conf_threshold: float = 0.5) -> List:
        """Filter detections by confidence threshold.

        Args:
            detections: Raw detection array
            conf_threshold: Minimum confidence threshold

        Returns:
            Filtered detections as list of (class_id, confidence, bbox)
        """
        if len(detections) == 0:
            return []

        detections = np.array(detections)
        filtered = []

        for det in detections:
            if len(det) >= 6:
                cls_id, conf, x1, y1, x2, y2 = det[:6]
                if conf >= conf_threshold:
                    filtered.append((int(cls_id), float(conf), [float(x1), float(y1), float(x2), float(y2)]))

        if self.debug:
            print(f"\n=== DETECTIONS ===")
            print(f"Raw detections: {len(detections)}")
            print(f"Filtered detections (conf >= {conf_threshold}): {len(filtered)}")

            for i, (cls_id, conf, bbox) in enumerate(filtered):
                class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                print(f"  {i+1}. {class_name} (conf={conf:.3f}) bbox={bbox}")

        return filtered

    def scale_bboxes_to_feature_space(self, detections: List, feature_map_shape: Tuple) -> List:
        """Scale bounding box coordinates from image space to feature map space.

        Args:
            detections: List of (class_id, confidence, bbox) tuples
            feature_map_shape: Shape of feature map [batch, channels, height, width]

        Returns:
            Detections with scaled bounding boxes
        """
        if len(detections) == 0:
            return []

        # Calculate scale factors
        _, _, feat_h, feat_w = feature_map_shape
        scale_x = feat_w / self.input_size
        scale_y = feat_h / self.input_size

        if self.debug:
            print(f"\n=== BBOX SCALING ===")
            print(f"Input size: {self.input_size}x{self.input_size}")
            print(f"Feature map size: {feat_h}x{feat_w}")
            print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")

        scaled_detections = []
        for cls_id, conf, bbox in detections:
            x1, y1, x2, y2 = bbox

            # Scale to feature map coordinates
            feat_x1 = x1 * scale_x
            feat_y1 = y1 * scale_y
            feat_x2 = x2 * scale_x
            feat_y2 = y2 * scale_y

            # Clamp to feature map bounds
            feat_x1 = max(0, min(feat_w - 1, feat_x1))
            feat_y1 = max(0, min(feat_h - 1, feat_y1))
            feat_x2 = max(0, min(feat_w - 1, feat_x2))
            feat_y2 = max(0, min(feat_h - 1, feat_y2))

            scaled_bbox = [feat_x1, feat_y1, feat_x2, feat_y2]
            scaled_detections.append((cls_id, conf, bbox, scaled_bbox))

            if self.debug:
                class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                print(f"  {class_name}: {bbox} -> {scaled_bbox}")

        return scaled_detections

    def extract_roi_features(self, feature_map: np.ndarray, scaled_detections: List) -> List:
        """Extract Region of Interest (RoI) features from feature map.

        Args:
            feature_map: Feature map tensor [batch, channels, height, width]
            scaled_detections: Detections with scaled bounding boxes

        Returns:
            List of (detection_info, roi_features) tuples
        """
        if len(scaled_detections) == 0:
            return []

        if self.debug:
            print(f"\n=== ROI EXTRACTION ===")

        # Remove batch dimension
        feature_map = feature_map[0]  # [channels, height, width]
        channels, feat_h, feat_w = feature_map.shape

        roi_features = []
        for i, (cls_id, conf, orig_bbox, scaled_bbox) in enumerate(scaled_detections):
            feat_x1, feat_y1, feat_x2, feat_y2 = scaled_bbox

            # Convert to integer coordinates
            x1, y1 = int(feat_x1), int(feat_y1)
            x2, y2 = int(feat_x2), int(feat_y2)

            # Ensure valid region
            if x2 <= x1 or y2 <= y1:
                if self.debug:
                    print(f"  ⚠️  ROI {i+1}: Invalid region [{x1}:{x2}, {y1}:{y2}]")
                continue

            # Extract ROI from feature map
            roi = feature_map[:, y1:y2, x1:x2]  # [channels, roi_h, roi_w]

            detection_info = {
                'class_id': cls_id,
                'confidence': conf,
                'original_bbox': orig_bbox,
                'scaled_bbox': scaled_bbox,
                'roi_shape': roi.shape
            }

            roi_features.append((detection_info, roi))

            if self.debug:
                class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                print(f"  ROI {i+1} ({class_name}): region=[{x1}:{x2}, {y1}:{y2}] shape={roi.shape}")

        return roi_features

    def apply_global_average_pooling(self, roi_features: List) -> List:
        """Apply Global Average Pooling (GAP) to convert ROI features to embeddings.

        Args:
            roi_features: List of (detection_info, roi_features) tuples

        Returns:
            List of (detection_info, embedding) tuples
        """
        if len(roi_features) == 0:
            return []

        if self.debug:
            print(f"\n=== GLOBAL AVERAGE POOLING ===")

        embeddings = []
        for i, (detection_info, roi) in enumerate(roi_features):
            # Apply GAP: average across spatial dimensions (height, width)
            embedding = np.mean(roi, axis=(1, 2))  # [channels] -> 1D vector

            embeddings.append((detection_info, embedding))

            if self.debug:
                class_name = COCO_CLASSES[detection_info['class_id']] if detection_info['class_id'] < len(COCO_CLASSES) else f"class_{detection_info['class_id']}"
                print(f"  Embedding {i+1} ({class_name}): {roi.shape} -> {embedding.shape}")
                print(f"    Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                print(f"    Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")

        return embeddings

    def l2_normalize_embeddings(self, embeddings: List) -> List:
        """L2 normalize embeddings for stable cosine distance calculations.

        Args:
            embeddings: List of (detection_info, embedding) tuples

        Returns:
            List of (detection_info, normalized_embedding) tuples
        """
        if len(embeddings) == 0:
            return []

        if self.debug:
            print(f"\n=== L2 NORMALIZATION ===")

        normalized_embeddings = []
        for i, (detection_info, embedding) in enumerate(embeddings):
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_embedding = embedding / norm
            else:
                normalized_embedding = embedding

            normalized_embeddings.append((detection_info, normalized_embedding))

            if self.debug:
                class_name = COCO_CLASSES[detection_info['class_id']] if detection_info['class_id'] < len(COCO_CLASSES) else f"class_{detection_info['class_id']}"
                print(f"  Embedding {i+1} ({class_name}): norm={norm:.4f} -> {np.linalg.norm(normalized_embedding):.4f}")

        return normalized_embeddings

    def visualize_embeddings(self, embeddings: List, save_path: str = None):
        """Visualize embeddings using dimensionality reduction techniques.

        Args:
            embeddings: List of (detection_info, embedding) tuples
            save_path: Optional path to save visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available, skipping visualization")
            return

        if len(embeddings) == 0:
            print("⚠️  No embeddings to visualize")
            return

        # Extract embedding vectors and labels
        vectors = np.array([emb for _, emb in embeddings])
        labels = [COCO_CLASSES[info['class_id']] if info['class_id'] < len(COCO_CLASSES)
                 else f"class_{info['class_id']}" for info, _ in embeddings]

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Embedding magnitude distribution
        norms = [np.linalg.norm(emb) for _, emb in embeddings]
        axes[0].bar(range(len(norms)), norms)
        axes[0].set_title("Embedding L2 Norms")
        axes[0].set_xlabel("Detection Index")
        axes[0].set_ylabel("L2 Norm")

        # Plot 2: Embedding heatmap (first 50 dimensions)
        if vectors.shape[1] > 50:
            vectors_vis = vectors[:, :50]
        else:
            vectors_vis = vectors

        # Create simple heatmap without seaborn
        im = axes[1].imshow(vectors_vis, cmap='RdBu_r', aspect='auto')
        axes[1].set_title(f"Embedding Heatmap (first {vectors_vis.shape[1]} dims)")
        axes[1].set_xlabel("Embedding Dimension")
        axes[1].set_ylabel("Detection")

        # Add colorbar
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved embedding visualization: {save_path}")
        else:
            plt.show()

    def compute_similarity_matrix(self, embeddings: List) -> np.ndarray:
        """Compute pairwise cosine similarity matrix between embeddings.

        Args:
            embeddings: List of (detection_info, embedding) tuples

        Returns:
            Similarity matrix
        """
        if len(embeddings) <= 1:
            return np.array([[1.0]] if len(embeddings) == 1 else [])

        vectors = np.array([emb for _, emb in embeddings])

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(vectors, vectors.T)

        if self.debug:
            print(f"\n=== SIMILARITY MATRIX ===")
            print(f"Matrix shape: {similarity_matrix.shape}")
            print("Similarity matrix:")
            for i in range(len(embeddings)):
                row_str = " ".join(f"{similarity_matrix[i,j]:.3f}" for j in range(len(embeddings)))
                class_name = COCO_CLASSES[embeddings[i][0]['class_id']] if embeddings[i][0]['class_id'] < len(COCO_CLASSES) else f"class_{embeddings[i][0]['class_id']}"
                print(f"  {i} ({class_name}): [{row_str}]")

        return similarity_matrix

    def save_results(self, embeddings: List, similarity_matrix: np.ndarray, output_path: str):
        """Save embeddings and analysis results to JSON file.

        Args:
            embeddings: List of (detection_info, embedding) tuples
            similarity_matrix: Pairwise similarity matrix
            output_path: Path to save results
        """
        results = {
            "model_path": self.model_path,
            "feature_map_name": self.feature_map_name,
            "feature_map_shape": list(self.feature_map_shape) if self.feature_map_shape else None,
            "num_detections": len(embeddings),
            "embedding_dimension": len(embeddings[0][1]) if embeddings else 0,
            "detections": [],
            "similarity_matrix": similarity_matrix.tolist() if similarity_matrix.size > 0 else []
        }

        for i, (detection_info, embedding) in enumerate(embeddings):
            class_name = COCO_CLASSES[detection_info['class_id']] if detection_info['class_id'] < len(COCO_CLASSES) else f"class_{detection_info['class_id']}"

            det_result = {
                "detection_id": i,
                "class_id": detection_info['class_id'],
                "class_name": class_name,
                "confidence": detection_info['confidence'],
                "original_bbox": detection_info['original_bbox'],
                "scaled_bbox": detection_info['scaled_bbox'],
                "roi_shape": list(detection_info['roi_shape']),
                "embedding": embedding.tolist(),
                "embedding_norm": float(np.linalg.norm(embedding))
            }
            results["detections"].append(det_result)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 Saved results to: {output_path}")

    def draw_detections_with_embeddings(self, image: np.ndarray, embeddings: List, save_path: str):
        """Draw detections on image with embedding information.

        Args:
            image: Original image
            embeddings: List of (detection_info, embedding) tuples
            save_path: Path to save annotated image
        """
        result_img = image.copy()

        # Color palette for different classes
        np.random.seed(42)
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                  for _ in range(len(COCO_CLASSES))]

        for i, (detection_info, embedding) in enumerate(embeddings):
            x1, y1, x2, y2 = [int(coord) for coord in detection_info['original_bbox']]
            class_id = detection_info['class_id']
            confidence = detection_info['confidence']

            # Choose color
            color = colors[class_id] if class_id < len(colors) else (255, 255, 255)

            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

            # Create label with embedding info
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            embedding_norm = np.linalg.norm(embedding)
            label = f"{class_name} {confidence:.2f} norm={embedding_norm:.2f}"

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save result
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        cv2.imwrite(save_path, result_img)
        print(f"🖼️  Saved annotated image: {save_path}")

    def process_image(self, image_path: str, conf_threshold: float = 0.5,
                     output_dir: str = "output/reid") -> List:
        """Main processing pipeline to generate Re-ID embeddings from an image.

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            output_dir: Directory to save outputs

        Returns:
            List of (detection_info, embedding) tuples
        """
        print(f"🚀 Processing image: {image_path}")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   Output directory: {output_dir}")

        # Step 1: Preprocess image
        original_image, input_feed = self.preprocess_image(image_path)

        # Step 2: Run inference
        detections, feature_map = self.run_inference(input_feed)

        # Step 3: Filter detections
        filtered_detections = self.filter_detections(detections, conf_threshold)

        if len(filtered_detections) == 0:
            print("⚠️  No detections found above confidence threshold!")
            return []

        # Step 4: Scale bounding boxes to feature map space
        scaled_detections = self.scale_bboxes_to_feature_space(filtered_detections, feature_map.shape)

        # Step 5: Extract RoI features
        roi_features = self.extract_roi_features(feature_map, scaled_detections)

        # Step 6: Apply Global Average Pooling
        embeddings = self.apply_global_average_pooling(roi_features)

        # Step 7: L2 normalize embeddings
        normalized_embeddings = self.l2_normalize_embeddings(embeddings)

        # Analysis and visualization
        similarity_matrix = self.compute_similarity_matrix(normalized_embeddings)

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save results JSON
        results_path = os.path.join(output_dir, f"{base_name}_reid_results.json")
        self.save_results(normalized_embeddings, similarity_matrix, results_path)

        # Save annotated image
        image_path = os.path.join(output_dir, f"{base_name}_reid_detections.jpg")
        self.draw_detections_with_embeddings(original_image, normalized_embeddings, image_path)

        # Save embedding visualization
        viz_path = os.path.join(output_dir, f"{base_name}_reid_embeddings.png")
        self.visualize_embeddings(normalized_embeddings, viz_path)

        print(f"\n✅ Processing complete! Generated {len(normalized_embeddings)} Re-ID embeddings")
        return normalized_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Generate Re-ID embeddings from RT-DETR with backbone features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="output/rtdetrv3_r18vd_6x_backbone.onnx",
                       help="Path to ONNX model with backbone features")
    parser.add_argument("--image", default="demo/demo.jpg",
                       help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--output", default="output/reid",
                       help="Output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("Please make sure you have exported the model with backbone features.")
        return

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return

    try:
        # Initialize generator
        generator = ReIDEmbeddingGenerator(args.model, debug=args.debug)

        # Process image
        embeddings = generator.process_image(
            args.image,
            conf_threshold=args.conf,
            output_dir=args.output
        )

        if embeddings:
            print(f"\n🎯 Summary:")
            print(f"   Generated {len(embeddings)} Re-ID embeddings")
            print(f"   Embedding dimension: {len(embeddings[0][1])}")
            print(f"   Results saved to: {args.output}")

            # Print embedding summary
            for i, (detection_info, embedding) in enumerate(embeddings):
                class_name = COCO_CLASSES[detection_info['class_id']] if detection_info['class_id'] < len(COCO_CLASSES) else f"class_{detection_info['class_id']}"
                norm = np.linalg.norm(embedding)
                print(f"   Embedding {i+1}: {class_name} (conf={detection_info['confidence']:.3f}, norm={norm:.3f})")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
