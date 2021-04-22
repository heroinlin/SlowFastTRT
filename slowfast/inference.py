#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import sys
import torch
import cv2
import numpy as np
import time
import struct
import json
from .config.defaults import get_cfg
from .utils import checkpoint as cu
from .models import build_model


class Inference(object):
    """
    Perform inference on the model.
    Args:
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
    """

    def __init__(self, cfg, device="cuda:0"):
        self.cfg = cfg
        self.input_video = cfg.DEMO.INPUT_VIDEO
        self.model = None
        self.half = False
        self.load_model()
        self.device = device
        if self.device != "cpu":
            self.model = self.model.cuda()
        self.inputs = []
        self.class_names_map = {}

    def load_model(self):
        self.model = build_model(self.cfg)
        self.model.eval()
        cu.load_test_checkpoint(self.cfg, self.model)
        if self.half:
            self.model.half()

    def _preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame, (self.cfg.DATA.TEST_CROP_SIZE, self.cfg.DATA.TEST_CROP_SIZE))
        frame = (frame / 255.0 - self.cfg.DATA.MEAN) / self.cfg.DATA.STD
        return frame

    def get_frames(self, valid_frames, indices):
        frames = []
        begin_index = indices[0]
        end_index = indices[-1]
        for index, frame in enumerate(valid_frames[begin_index:end_index+1]):
            if (index + begin_index) in indices:
                frames.append(self._preprocess(frame))
            if (index + begin_index) > end_index:
                break
        return frames

    def get_input(self, frames):
        first_pathway = torch.from_numpy(
            np.array(frames)[::8, :, :, :]).float().permute(3, 0, 1, 2).unsqueeze(0)
        socond_pathway = torch.from_numpy(
            np.array(frames)).float().permute(3, 0, 1, 2).unsqueeze(0)
        if self.half:
            first_pathway = first_pathway.half()
            socond_pathway = socond_pathway.half()
        if self.device != "cpu":
            first_pathway = first_pathway.cuda()
            socond_pathway = socond_pathway.cuda()
        # print(first_pathway.size(), socond_pathway.size())
        net_input = [first_pathway, socond_pathway]
        return net_input

    def get_inputs(self):
        video = cv2.VideoCapture()
        if not video.open(self.input_video):
            print("open video error!\n")
            self.inputs = []
            return -100
        frames = []
        valid_frames = []
        # 对视频均匀间隔取指定数量帧
        video_frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        if video_fps <= 1:  # if get fps error, set fps default is 25.
            video_fps = 25
        frame_num = 0
        while True:
            ret, frame = video.read()
            if ret:
                valid_frames.append(frame)
            frame_num += 1
            if frame_num >= video_frame_count:
                break

        valid_frame_count = len(valid_frames)
        video_time = valid_frame_count / video_fps
        if valid_frame_count < 32: # valid frames need >= 32
            print("video frome num < 32!\n")
            self.inputs = []
            return -1
        if 2.0 <= video_time < 6.0:
            indices = [int((i) * max(1.0, ((valid_frame_count - 1) / self.cfg.DATA.NUM_FRAMES)))
                       for i in range(self.cfg.DATA.NUM_FRAMES)]
            frames = self.get_frames(valid_frames, indices)
            self.inputs = self.get_input(frames)
        elif video_time < 2.0:
            print("video time is too short!\n")
            self.inputs = []
            return -1
        else:
            video_clip_num = round((video_time - 0.5) / 4.0)
            self.inputs = []
            for video_clip_item in range(video_clip_num):
                start_frame_index = int(video_clip_item * video_fps * 4.0)
                end_frame_index = min(
                    int((video_clip_item+1) * video_fps * 4.0), valid_frame_count)
                valid_frames_clip = valid_frames[start_frame_index: end_frame_index]
                indices = [int((i) * max(1.0, ((len(valid_frames_clip) - 1) / self.cfg.DATA.NUM_FRAMES)))
                           for i in range(self.cfg.DATA.NUM_FRAMES)]
                frames = self.get_frames(valid_frames_clip, indices)
                self.inputs.append(self.get_input(frames))
        video.release()
        return 0

    def forward(self):
        predicts = []
        if not len(self.inputs):
            print("get input error! please check it!")
            return predicts
        if isinstance(self.inputs[0], list):
            pred_array = np.zeros([0, len(self.class_names_map)], np.float)
            index = 0
            for inputs in self.inputs:
                index += 1
                preds = self.model(inputs)
                preds = preds.cpu().data.numpy()
                pred_array = np.vstack((pred_array, preds[0]))
            preds = np.max(pred_array, axis=0, keepdims=1)
        else:
            preds = self.model(self.inputs)
            preds = preds.cpu().data.numpy()
        pred_class_id_list = list(np.argsort(-preds[0]))
        class_names_list = list(self.class_names_map.keys())
        # print(class_names_list)
        for class_name_id in pred_class_id_list:
            # predicts.append({class_name: float(preds[0][self.class_names_map[class_name]])})
            predicts.append({"name": class_names_list[class_name_id],
                             "score": round(float(preds[0][class_name_id]), 2)})
        return predicts


def inference_demo(cfg):
    demo = Inference(cfg)
    print("load model success!\n")
    start_time = time.time()
    demo.get_inputs()
    predicts = demo.forward()
    print(predicts)
    end_time = time.time()
    print(
        f"load video and inference cost time: {end_time - start_time:.03f} s")
