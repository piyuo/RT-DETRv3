# Changelog

## 0.1.0 (2025-08-30)


### Features

* add 'op_types_to_quantize' option for Conv layers in ONNX model optimization ([9be65df](https://github.com/piyuo/RT-DETRv3/commit/9be65df31d486b2e4cd39f9d696a8f2fed86a7a9))
* add calibration dataset creation and quantization scripts, update .gitignore for new paths ([823d1e6](https://github.com/piyuo/RT-DETRv3/commit/823d1e6a21fc82b6d4a8d83e3dcdc173c5268e50))
* add compress.py for model compression using AutoCompression with COCO dataset integration ([e7575b7](https://github.com/piyuo/RT-DETRv3/commit/e7575b7b3be219bcefaeaf9f16a4cc77f4ab929d))
* add logging for dataset loading and processing in COCO detection ([df7f707](https://github.com/piyuo/RT-DETRv3/commit/df7f707b6641f4722865cd39166a6914fbc7800b))
* add onnx export support ([4aef9b2](https://github.com/piyuo/RT-DETRv3/commit/4aef9b2b44e9e98e0e8b51a484e55202ef88f05d))
* add ONNX model export and loading scripts with shape fixing ([da359d9](https://github.com/piyuo/RT-DETRv3/commit/da359d92e4530fdd1a7ec374494f84801deaa57c))
* add optimize_onnx_model.py for ONNX model optimization with Olive ([7b9b778](https://github.com/piyuo/RT-DETRv3/commit/7b9b77855755ab9be3ed5d141e97eda8f141278b))
* add scripts for calibration image processing and remove unused dataset generation script ([0511705](https://github.com/piyuo/RT-DETRv3/commit/0511705ec2375223d688f3d8d85d9ba2d4122643))
* enhance ONNX model optimization with additional quantization options and debugging features ([37b19be](https://github.com/piyuo/RT-DETRv3/commit/37b19be2a5d6b1d10dddb271a5b68fb27ff831ca))
* enhance ONNX model optimization with new data configuration and preprocessing scripts ([992bcc2](https://github.com/piyuo/RT-DETRv3/commit/992bcc2fa76d768067b9035076747599088ff062))
* refactor compress.py to define dataloader creation as a function and update eval_callback to use it ([9fc1ca0](https://github.com/piyuo/RT-DETRv3/commit/9fc1ca049ac74ba7a3f793ed2ac1ae4dd1e73514))
* refine compress.py by correcting import statement, enhancing data loader configuration, and simplifying evaluation callback setup ([4da026f](https://github.com/piyuo/RT-DETRv3/commit/4da026f901f8cfdd07247b9b4f279b3eaeaaaa7d))
* update .gitignore to include coco dataset paths and add make_calibration_640.py and optimize_model.py scripts ([9875b88](https://github.com/piyuo/RT-DETRv3/commit/9875b88b7c26056bb8a950a90beda03fc062d5a5))
* update COCO validation dataset configuration and add utility for dataset reduction ([9043434](https://github.com/piyuo/RT-DETRv3/commit/90434347cdbf957ed694b3ea6d6518b3f308b4a1))
* update compress.py to correct import statement, redefine model filenames, and simplify evaluation callback setup ([9822198](https://github.com/piyuo/RT-DETRv3/commit/98221987e2107b2d26dbe4c37ad7b6ee142fc0b1))
* update environment setup and scripts for ONNX model optimization with Olive ([3a92c1b](https://github.com/piyuo/RT-DETRv3/commit/3a92c1b7b37652773c9a5b9ff30b278951af0218))
* update export_onnx.sh to remove unnecessary export_onnx=True flag, enhance init_env.sh with additional package installations, and delete unused qat.py script ([94805da](https://github.com/piyuo/RT-DETRv3/commit/94805da64c2dcb82344e9c0057a6bc12ef9633c4))
* update ONNX model optimization scripts and add dataset testing functionality ([3179100](https://github.com/piyuo/RT-DETRv3/commit/317910058cf9fb134b318f6f8a51755f6fc9de14))
* update ONNX model optimization scripts and configuration for Olive integration ([17e4f61](https://github.com/piyuo/RT-DETRv3/commit/17e4f6155776aec6ee955c07a9487dbf26eea6d3))
* update ONNX quantization configuration with new activation type and per_channel settings ([8a606ad](https://github.com/piyuo/RT-DETRv3/commit/8a606ad421dfc6fe68569dd9c80821650abe5c01))


### Bug Fixes

* uncomment export model command and update ONNX save file path ([aaec741](https://github.com/piyuo/RT-DETRv3/commit/aaec7418569afc75f420ed259bfdd999a849ee35))
