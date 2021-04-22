# -*- coding:utf-8 -*-
import os
import sys
from collections import OrderedDict
import torch
import argparse
work_root = os.path.split(os.path.realpath(__file__))[0]
from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        default=os.path.join(
            work_root, "configs/SLOWFAST_4x16_R50_inference.yaml"),
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
        default=os.path.join(work_root,
                             "weights/checkpoint_epoch_00050.pyth"),
        help='test model file path',
    )
    parser.add_argument(
        '--save',
        type=str,
        default=os.path.join(work_root, "weights/checkpoint_epoch_00050.onnx"),
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
    print(cfg.DATA)
    print("export pytorch model to onnx!\n")
    device = "cuda:0"
    with torch.no_grad():
        model = build_model(cfg)
        model = model.to(device)
        model.eval()
        cu.load_test_checkpoint(cfg, model)
        if half_flag:
            model.half()
        # with open(save_checkpoint_file, 'wb') as file:
        #     torch.save({"model_state": model.state_dict()}, file)
        fast_pathway= torch.randn(1, 3, 32, 256, 256)
        slow_pathway= torch.randn(1, 3, 4, 256, 256)
        fast_pathway = fast_pathway.to(device)
        slow_pathway = slow_pathway.to(device)
        inputs = [slow_pathway, fast_pathway]
        for p in model.parameters():
            p.requires_grad = False

        torch.onnx.export(model, inputs, save_checkpoint_file, opset_version=12)
        onnx_check()


def onnx_check():
    import onnx
    args = parser_args()
    print(args)
    onnx_model_path = args.save
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)


if __name__ == '__main__':
    main()
