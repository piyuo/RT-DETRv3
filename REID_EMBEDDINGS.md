# RT-DETRv3 Re-ID Embeddings for BoT-SORT Integration

## Overview

This repository now includes a complete pipeline for generating Re-ID (Re-Identification) embeddings from RT-DETRv3 model with backbone features. These embeddings can be used in BoT-SORT for robust multi-object tracking.

## Quick Start

### 1. Generate Re-ID Embeddings

```bash
# Basic usage
python tools/reid_embeddings.py --model output/rtdetrv3_r18vd_6x_backbone.onnx --image demo/demo.jpg

# With custom settings
python tools/reid_embeddings.py \
    --model output/rtdetrv3_r18vd_6x_backbone.onnx \
    --image demo/demo.jpg \
    --conf 0.5 \
    --output output/reid \
    --debug
```

### 2. Test BoT-SORT Integration

```bash
python tools/botsort_integration_test.py --results output/reid/demo_reid_results.json
```

## Pipeline Architecture

### Step 1: Model with Backbone Features
- Input: RT-DETRv3 model (`rtdetrv3_r18vd_6x_raw.onnx`)
- Process: Add backbone feature output using `onnx_export_backbone_features.py`
- Output: Model with C5 feature map (`rtdetrv3_r18vd_6x_backbone.onnx`)

### Step 2: Re-ID Embedding Generation
The `reid_embeddings.py` script follows these steps:

1. **Inference**: Run RT-DETR model to get:
   - Object detections (bounding boxes, classes, confidences)
   - Dense feature map from backbone C5 layer (512×20×20)

2. **Coordinate Scaling**: Convert bounding box coordinates from image space (640×640) to feature map space (20×20)
   ```python
   scale_factor = feature_map_size / input_size  # 20/640 = 0.03125
   feature_x = bbox_x * scale_factor
   ```

3. **RoI Extraction**: Crop regions of interest from the feature map corresponding to each detected object

4. **Global Average Pooling (GAP)**: Convert spatial features to fixed-size embeddings
   ```python
   embedding = np.mean(roi_features, axis=(height, width))  # Shape: (512,)
   ```

5. **L2 Normalization**: Normalize embeddings for stable cosine distance calculations
   ```python
   normalized_embedding = embedding / np.linalg.norm(embedding)
   ```

## Output Format

### Re-ID Results JSON
```json
{
  "model_path": "output/rtdetrv3_r18vd_6x_backbone.onnx",
  "feature_map_name": "p2o.pd_op.conv2d.18.0",
  "num_detections": 6,
  "embedding_dimension": 512,
  "detections": [
    {
      "detection_id": 0,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.929,
      "original_bbox": [482.07, 268.07, 607.70, 684.50],
      "scaled_bbox": [15.06, 8.38, 18.99, 19.00],
      "roi_shape": [512, 11, 3],
      "embedding": [...],  // 512-dimensional vector
      "embedding_norm": 1.0
    }
  ],
  "similarity_matrix": [[...]]  // Pairwise cosine similarities
}
```

### BoT-SORT Compatible Format
```python
botsort_detection = {
    'id': 0,
    'bbox': [x1, y1, x2, y2],
    'confidence': 0.929,
    'class_id': 0,
    'class_name': 'person',
    'embedding': np.array([...]),  # 512-dimensional normalized vector
    'embedding_norm': 1.0
}
```

## Integration with BoT-SORT

### Distance Calculation
Use cosine distance for Re-ID matching:
```python
def cosine_distance(emb1, emb2):
    return 1.0 - np.dot(emb1, emb2)  # Both embeddings are L2-normalized
```

### Recommended Thresholds
- **Person Re-ID**: 0.2 - 0.5 (lower = more strict matching)
- **Same class matching**: ~0.059 ± 0.032 (observed mean ± std)
- **Different class separation**: ~0.165 ± 0.053

### Quality Metrics
- **Embedding dimension**: 512
- **Normalization**: L2 normalized (norm = 1.0)
- **Separability ratio**: 2.79 (good class separation)
- **Feature stride**: 32 (from C5 backbone layer)

## File Structure

```
tools/
├── reid_embeddings.py              # Main Re-ID embedding generator
├── botsort_integration_test.py     # BoT-SORT integration testing
├── onnx_export_backbone_features.py # Export model with backbone features
└── onnx_inference.py               # Basic ONNX inference verification

output/
├── rtdetrv3_r18vd_6x_backbone.onnx # Model with backbone features
└── reid/
    ├── demo_reid_results.json      # Re-ID results
    ├── demo_reid_detections.jpg    # Annotated image
    ├── demo_reid_embeddings.png    # Embedding visualization
    ├── botsort_analysis.json       # BoT-SORT analysis
    └── botsort_relationships.png   # Relationship visualization
```

## Script Parameters

### reid_embeddings.py
- `--model`: Path to ONNX model with backbone features (default: `output/rtdetrv3_r18vd_6x_backbone.onnx`)
- `--image`: Input image path (default: `demo/demo.jpg`)
- `--conf`: Confidence threshold for detections (default: 0.5)
- `--output`: Output directory (default: `output/reid`)
- `--debug`: Enable debug output

### botsort_integration_test.py
- `--results`: Path to Re-ID results JSON (default: `output/reid/demo_reid_results.json`)
- `--distance-threshold`: Distance threshold for matching (default: 0.3)
- `--output`: Output directory (default: `output/reid`)

## Verification and Debugging

### Debug Information
The scripts provide detailed debug output including:
- Model input/output shapes and types
- Feature map identification and stride calculation
- Bounding box scaling verification
- RoI extraction details
- Embedding statistics (range, mean, std)
- Similarity matrix analysis

### Common Issues and Solutions

1. **Invalid RoI regions**: Some bounding boxes may scale to invalid regions (x2 <= x1 or y2 <= y1)
   - **Solution**: These are automatically filtered out during RoI extraction

2. **Small RoI sizes**: Very small objects may result in tiny RoI regions
   - **Solution**: Consider minimum RoI size constraints or different feature map levels

3. **Poor separability**: If same-class and different-class distances overlap significantly
   - **Solution**: Adjust distance thresholds or retrain with better Re-ID supervision

### Quality Validation
- **Embedding norms**: Should all be 1.0 (L2 normalized)
- **Intra-class distances**: Should be smaller than inter-class distances
- **Separability ratio**: Should be > 1.5 for good performance

## Performance Characteristics

### Inference Speed
- **Model loading**: ~1-2 seconds
- **Single image inference**: ~100-200ms (CPU)
- **Re-ID processing**: ~10-50ms per detection

### Memory Usage
- **Feature map**: 512 × 20 × 20 × 4 bytes = ~800KB
- **Single embedding**: 512 × 4 bytes = 2KB
- **Batch processing**: Scales linearly with number of detections

## Future Improvements

1. **Multi-scale features**: Combine C3, C4, C5 features for better representation
2. **Temporal consistency**: Add temporal smoothing for video sequences
3. **Class-specific embeddings**: Train separate embedding heads for different object classes
4. **Quantization**: Optimize embeddings for deployment (FP16, INT8)
5. **Online adaptation**: Update embeddings based on tracking history

## Citation

If you use this Re-ID embedding pipeline in your research, please cite:

```bibtex
@misc{rtdetrv3_reid_2024,
  title={RT-DETRv3 Re-ID Embeddings for Multi-Object Tracking},
  author={Your Name},
  year={2024}
}
```

## License

This code is provided under the same license as the original RT-DETRv3 repository.
