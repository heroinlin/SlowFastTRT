"""

This module implements the SlowFastTRT class.
"""


import ctypes
import os
import numpy as np
import cv2
import random
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading
import time
working_root = os.path.split(os.path.realpath(__file__))[0]


class TrtInference(object):
    def __init__(self, model_path=None, cuda_ctx=None):
        self._model_path = model_path
        if self._model_path is None:
            print("please set trt model path!")
            exit()
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx is None:
            self.cuda_ctx = cuda.Device(0).make_context()
        if self.cuda_ctx:
            self.cuda_ctx.push()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def _load_plugins(self):
        pass

    def _load_engine(self):
        with open(self._model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings

    def destroy(self):
        """Free CUDA memories and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream
        if self.cuda_ctx:
            self.cuda_ctx.pop()
            del self.cuda_ctx

    def inference(self, inputs):
        np.copyto(self.host_inputs[0], inputs[0].ravel())
        np.copyto(self.host_inputs[1], inputs[1].ravel())
        if self.cuda_ctx:
            self.cuda_ctx.push()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        cuda.memcpy_htod_async(
            self.cuda_inputs[1], self.host_inputs[1], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        output = self.host_outputs[0]
        return output


class SlowFastTRT(TrtInference):
    """SlowFastTRT video classify."""
    _names = None
    _filters = None
    _img_size = 256
    _mean = [0.45, 0.45, 0.45]
    _std = [0.225, 0.225, 0.225]
    def __init__(self, model_path=None, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self._show = False
        super(SlowFastTRT, self).__init__(model_path, cuda_ctx)
        self.frame_num = 32
        self.sampling_rate = 3

    def set_config(self, key, value):
        if 'names' == key:
            self._names = value
        elif 'filters' == key:
            self._filters = value
        elif 'img_size' == key:
            self._img_size = value
        elif 'mean' == key:
            self._mean = value
        elif 'std' == key:
            self._std = value

    def _model_release(self):
        pass

    def _preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self._img_size, self._img_size))
        frame = (frame / 255.0 - self._mean) / self._std
        return frame

    def _preprocess_trt(self, filename):
        video = cv2.VideoCapture()
        if not video.open(filename):
            print("open video error!\n")
            inputs = []
            return
        frames = []
        frame_num = 0
        while len(frames) < self.frame_num:
            _, frame = video.read()
            if frame is None:
                break
            frame_num += 1
            if frame_num % self.sampling_rate == 0:
                frames.append(self._preprocess(frame))
                # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        video.release()
        first_pathway = np.array(frames)[::8, :, :, :].transpose(3, 0, 1, 2)
        first_pathway = np.expand_dims(first_pathway, axis=0)
        socond_pathway = np.array(frames).transpose(3, 0, 1, 2)
        socond_pathway = np.expand_dims(socond_pathway, axis=0)
        # print(first_pathway.shape, socond_pathway.shape)
        inputs = [first_pathway, socond_pathway]
        return inputs

    def _postprocess_trt(self, preds):
        """Postprocess TRT Yolov5 output."""
        pred_class_id_list = list(np.argsort(-preds))
        # print(pred_class_id_list)
        results = []
        for idx in pred_class_id_list:
            if self._filters is None:
                results.append({
                                "name":self._names[idx],
                                "score":round(float(preds[idx]), 2)})
            else:
                if self._names[idx] in self._filters:
                    results.append({
                                    "name":self._names[idx],
                                    "score":round(float(preds[idx]), 2)})
        return results

    def classify(self, filename=None):
        """Detect objects in the input image."""
        inputs = self._preprocess_trt(filename)
        pred = self.inference(inputs)
        output = self._postprocess_trt(pred)
        return output


class BehaviorSlowFastTRT(SlowFastTRT):
    """
        Behavior TRT Classifier
    """
    _names =[ "eat", "walk", "sit", "run", "talk", "watch", "play_ball"
              ]
    _filters =[ "run"]
    

def compute_time(func, args, run_num=100):
    start_time = time.time()
    for i in range(run_num):
        func(*args)
    end_time = time.time()
    avg_run_time = (end_time - start_time)*1000/run_num
    return avg_run_time


def compute_inference_time():
    engine_file_path = os.path.join(working_root, "trt_models/checkpoint_epoch_00050.engine")
    print("engine_file_path: ", engine_file_path)
    classifier = BehaviorSlowFastTRT(model_path=engine_file_path)
    input_video_paths = [os.path.join(working_root, "data/videos/1.flv")]
    input_video_path = input_video_paths[0]
    inputs = classifier._preprocess_trt(input_video_path)
    pred = classifier.inference(inputs)
    outputs = classifier._postprocess_trt(pred)
    print(outputs)
    preprocess_time = compute_time(classifier._preprocess_trt, [input_video_path])
    print("avg preprocess time is {:02f} ms".format(preprocess_time))

    inference_time = compute_time(classifier.inference, [inputs])
    print("avg inference time is {:02f} ms".format(inference_time))

    postprocess_time = compute_time(classifier._postprocess_trt, [pred])
    print("avg postprocess time is {:02f} ms".format(postprocess_time))

    total_time = compute_time(classifier.classify, [input_video_path])
    print("avg total predict time is {:02f} ms".format(total_time))

    classifier.destroy()


if __name__ == "__main__":
    compute_inference_time()
