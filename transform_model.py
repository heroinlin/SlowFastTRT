'''
Author: heroin
Date: 2020-10-22 09:17:28
LastEditTime: 2020-10-22 09:43:41
LastEditors: Please set LastEditors
Description: simplier model
FilePath: /ml/slowfast/transform_model.py
'''
#-*- coding:utf-8 -*-
import os
from collections import OrderedDict
import torch
import argparse

sys.path.insert(0, os.getcwd())
from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
work_root = os.path.split(os.path.realpath(__file__))[0]


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        default=os.path.join(work_root, "configs/SLOWFAST_4x16_R50_inference.yaml"),
        help="Path to the config file",
    )
    parser.add_argument(
        '--half',
        type=bool,
        default=False,
        help='use half mode',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=os.path.join(work_root, "checkpoints/checkpoint_epoch_00050.pyth"),
        help='test model file path',
    )
    parser.add_argument(
        '--save',
        type=str,
        default=os.path.join(work_root, "checkpoints/checkpoint_epoch_00050.pyth"),
        help='save model file path',
    )
    return parser.parse_args()


def main():
    args = parser_args()
    print(args)
    cfg_file = args.cfg_file
    checkpoint_file = args.checkpoint
    save_checkpoint_file = args.save
    half_flag = args.half
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_file

    print("simplifier model!\n")
    with torch.no_grad():
        model = build_model(cfg)
        model.eval()
        cu.load_test_checkpoint(cfg, model)
        if half_flag:
            model.half()
        with open(save_checkpoint_file, 'wb') as file:
            torch.save({"model_state": model.state_dict()}, file)


if __name__ == '__main__':
    main()
