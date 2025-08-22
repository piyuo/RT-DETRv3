# tools/compress.py

import os
import sys

# Get the path to the project's root directory and add it to sys.path
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_path)

import yaml
import paddle
from paddleslim.auto_compression import AutoCompression

# Import the reader and dataset from PaddleDetection
from ppdet.data.reader import EvalReader
from ppdet.data.source.coco import COCODataSet
from ppdet.core.workspace import load_config # Make sure this import is correct

# Import the new evaluation function from eval.py
from tools.eval_act import eval_function

paddle.enable_static()

def main():
    config_file = "configs/quant/rtdetr_quant_cfg.yml"
    with open(config_file, 'r') as f:
        all_config = yaml.safe_load(f)

    # 1. Get model paths and reader config from YAML
    global_config = all_config['Global']
    model_dir = global_config['model_dir']
    model_filename = global_config['model_filename']
    params_filename = global_config['params_filename']
    reader_config_file = global_config['reader_config']

    # 2. Load the evaluation reader configuration
    reader_cfg = yaml.safe_load(open(reader_config_file))['EvalReader']

    # 3. Prepare the dataset and data loader
    dataset_root = os.path.join("dataset", "coco")
    image_dir = reader_cfg['dataset']['image_dir']
    anno_path = reader_cfg['dataset']['anno_path']

    calib_dataset = COCODataSet(
        dataset_dir=dataset_root,
        image_dir=image_dir,
        anno_path=anno_path
    )

    # EvalReader will be the dataloader passed to AutoCompression
    train_dataloader = EvalReader(
        sample_transforms=reader_cfg['sample_transforms'],
        batch_size=reader_cfg['batch_size'],
        shuffle=reader_cfg['shuffle'],
        drop_last=reader_cfg['drop_last']
    )(dataset=calib_dataset, worker_num=0, return_list=True)

    # 4. Define the evaluation callback
    # We load the main configuration directly from the hardcoded path.
    # The previous code for ArgumentParser was incorrect and unnecessary.
    cfg = load_config(config_file)

    # The wrapper for the eval_function
    def eval_callback(program, place, exe, scope):
        # The dataloader is already prepared and passed to AutoCompression.
        # We just need to call our function with the provided arguments.
        # We also need to pass a mock FLAGS object or fix eval_function to not need it
        # For now, let's create a simple object to hold config
        class MockFLAGS:
            def __init__(self, config):
                self.config = config
                self.slice_infer = False

        mock_flags = MockFLAGS(config_file)

        return eval_function(program, place, exe, scope, train_dataloader, mock_flags, cfg)

    # 5. Run AutoCompression
    ac = AutoCompression(
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        save_dir="output/rtdetrv3_r18vd_6x_quant",
        config=all_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_callback
    )
    ac.compress()

if __name__ == "__main__":
    main()
