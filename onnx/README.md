# RT-DETRv3 Re-ID Embeddings for BoT-SORT Integration

## Overview

This folder contain pipeline for generating Re-ID (Re-Identification) embeddings from RT-DETRv3 model with backbone features. These embeddings can be used in BoT-SORT for multi-object tracking.

### Key Features

- **🔒 Robust Implementation**: Addresses all common failure modes and edge cases
- **🔍 Comprehensive Validation**: Built-in checks for model structure, detection format, and coordinate consistency
- **⚙️ Flexible Configuration**: Supports different preprocessing methods, detection layouts, and feature maps
- **📊 Quality Assurance**: Automatic embedding quality assessment and separability analysis
- **🛠️ Production-Ready**: Extensive error handling, logging, and validation

## Quick Start

### 1. Export Model with Backbone Features

```bash
# Auto-detect and export best backbone feature (C4), it use original model in onnx/input and convert it to onnx/backbone
onnx/export_backbone.sh
```

### 2. Validate Pipeline (Recommended), validate if C4 level enough, C4 recommend in most paper

```bash
# Run comprehensive validation
onnx/validate_reid_pipeline.sh
```

### 3. Test BoT-SORT Integration, use result from Validate Pipeline

```bash
onnx/botsort_integration_test.sh
```

### 4. Validate pedestrian reid embedding

```bash
onnx/validate_pedestrian.sh
```

### 5. Sample code to generate Re-ID Embeddings

```bash
onnx/reid_embeddings.sh
```

### 6. RT-DETRv3 quantization, from onnx/backbone -> onnx/output

```bash
onnx/quantize.sh
```

### 7. run inference on demo.jpg to validate the output model

```bash
onnx/inference.sh
```

### Common Issues and Solutions

#### Issue: "Poor class separability detected"

**Solution**: Consider multi-scale features or model fine-tuning, right now we already use C3 feature map, it can not get bigger, maybe bigger input image size will help, but current 640 is already slow for us, so we stuck on image size 640.

## File Structure (Updated)

```text
onnx/
├── botsort_integration_test.py                  # BoT-SORT Integration Test for RT-DETRv3 Re-ID Embeddings
├── botsort_integration_test.sh                  # Script to BoT-SORT Integration Test
├── export_backbone.py                           # Backbone feature exporter
├── export_backbone.sh                           # Script to run Backbone feature exporter
├── README.md                                    # Describe how to use these script files.
├── reid_embeddings.py                           # Re-ID generator
├── reid_embeddings.sh                           # Script to run Re-ID generator
├── validate_reid_pipeline.py                    # Comprehensive pipeline validator
├── validate_reid_pipeline.sh                    # Script to run Comprehensive pipeline validator
├── reid_usage_example.py                        # Usage examples
└── onnx_inference.py                            # Basic ONNX inference verification
├── input/
    ├── rtdetrv3_r18vd_6x.onnx                   # The RAW RT-DETRv3 model, convert from weight no optimization.
├── backbone/
    ├── rtdetrv3_r18vd_6x.onnx                   # The RT-DETRv3 model with feature backbone output.
├── output/
    ├── rtdetrv3_r18vd_6x.onnx                   # Final RT-DETRv3 model, Add feature backbone and optimization.
├── env/                                         # Python virtual env.
├── demo/
    ├── demo.jpg                                 # The demo image for validation and run inference.
├── validation/                                  # The validation files.

```

### Debug Information

The robust implementation provides detailed debug output including:

- Model input/output shapes and types
- Feature map identification and stride calculation
- Detection format validation with sample outputs
- Bounding box scaling verification with coordinate traces
- RoI extraction details with region validation
- Embedding statistics (range, mean, std, norm)
- Similarity matrix analysis with separability metrics
- Quality validation results

### Quality Validation

- **Embedding norms**: Should all be 1.0 (L2 normalized)
- **Intra-class distances**: Should be smaller than inter-class distances
- **Separability ratio**: Should be > 1.5 for good performance
- **Consistency**: Same image should produce nearly identical embeddings (>0.99 similarity)

## Performance Characteristics

### Inference Speed

- **Model loading**: ~1-2 seconds
- **Single image inference**: ~100-200ms (CPU)
- **Re-ID processing**: ~10-50ms per detection
- **Validation overhead**: ~50-100ms per run

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
6. **Batch processing**: Optimize for multi-image inference
7. **GPU acceleration**: CUDA/TensorRT optimization for real-time deployment

## License

This code is provided under the same license as the original RT-DETRv3 repository.
