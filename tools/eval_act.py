# tools/eval_act.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer, Trainer_ARSL, init_parallel_env
from ppdet.metrics.coco_utils import json_eval_results
from ppdet.slim import build_slim_model
from ppdet.utils.logger import setup_logger
from ppdet.utils.cli import ArgsParser, merge_args

logger = setup_logger('eval')


def eval_function(program, place, exe, scope, eval_dataloader, FLAGS, cfg):
    """
    Evaluation function for PaddleSlim's AutoCompression.

    Args:
        program (fluid.Program): The main program for evaluation.
        place (fluid.Place): The device to run on (e.g., CPU, GPU).
        exe (fluid.Executor): The executor for running the program.
        scope (fluid.Scope): The scope containing the model variables.
        eval_dataloader (paddle.io.DataLoader): The data loader for the evaluation dataset.
        FLAGS (object): Arguments passed from the command line.
        cfg (object): The configuration object loaded from YAML.

    Returns:
        float: The mean average precision (mAP) score.
    """

    # PaddleSlim passes the model and other components, so we don't need to load them here.
    # We just need to set up the trainer and run the evaluation.

    if FLAGS.json_eval:
        logger.info(
            "In json_eval mode, PaddleDetection will evaluate json files in "
            "output_eval directly.")
        json_eval_results(
            cfg.metric,
            json_directory=FLAGS.output_eval,
            dataset=create('EvalDataset')())
        return

    # init parallel environment if nranks > 1
    init_parallel_env()
    ssod_method = cfg.get('ssod_method', None)
    if ssod_method == 'ARSL':
        # build ARSL_trainer
        trainer = Trainer_ARSL(cfg, mode='eval')
        # load ARSL_weights
        trainer.load_weights(cfg.weights, ARSL_eval=True)
    else:
        # build trainer
        # Pass the program, executor, and scope provided by AutoCompression
        trainer = Trainer(
            cfg,
            mode='eval',
            eval_program=program,
            exe=exe,
            eval_scope=scope
        )
        # We don't need to load weights here, as AutoCompression handles it.

    # Set the dataloader for evaluation
    trainer.eval_dataloader = eval_dataloader

    # training
    if FLAGS.slice_infer:
        results = trainer.evaluate_slice(
            slice_size=FLAGS.slice_size,
            overlap_ratio=FLAGS.overlap_ratio,
            combine_method=FLAGS.combine_method,
            match_threshold=FLAGS.match_threshold,
            match_metric=FLAGS.match_metric)
    else:
        results = trainer.evaluate()

    # Return the mAP score, which is a common metric for object detection.
    # The exact key may vary, but 'bbox_mAP' is typical for COCO.
    return results.get('bbox_mAP', 0.0)

def main():
    FLAGS = ArgsParser().parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # Set up device
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    place = paddle.set_device('gpu') if cfg.use_gpu else paddle.set_device('cpu')

    # If slim_config is provided, build the slim model.
    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='eval')

    # Basic checks
    from ppdet.utils.check import check_config, check_gpu, check_version
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    # Create the main program and executor
    exe = paddle.static.Executor(place)
    scope = paddle.static.Scope()

    # This is a basic example, in a real scenario you would have a more complex
    # setup with `create_eval_program`. For this wrapper, we just need to pass
    # the necessary objects to the `eval_function`.

    # Placeholder for eval_dataloader, as it will be provided by AutoCompression
    eval_dataloader = None

    # Run the evaluation with a dummy program and scope
    eval_function(paddle.static.Program(), place, exe, scope, eval_dataloader, FLAGS, cfg)


if __name__ == '__main__':
    main()
