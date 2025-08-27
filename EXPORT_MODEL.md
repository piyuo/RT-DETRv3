# Export weight to ONNX/NCNN

this file describe how to export downloaded weight file to ONNX/NCNN format,
before start , run init script on prepare python environment.

## init python environment

```bash
  scripts/init_env.sh
```

## Export to Onnx

put download file "rtdetrv3_r18vd_6x.pdparams" to /weights folder and run script,
it will generate

1. PaddlePaddle model (raw file, unoptimized)
/output/paddlepaddle/model.json
/output/paddlepaddle/model.pdiparams

2. Onnx model (raw file, unoptimized)
/output/rtdetrv3_r18vd_6x_raw.onnx

```bash
  scripts/export_model.sh
```

## Quantize onnx model

run script, it will generate the quantized model at
/output/rtdetrv3_r18vd_6x.onnx

```bash
  scripts/onnx_quantize.sh
```
