import os
import argparse
import tensorrt as trt
from onnx import ModelProto
work_root = os.path.split(os.path.realpath(__file__))[0]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
"""SlowFast onnx模型转trt模型"""


def build_engine_trt8(onnx_path, shapes, precision_mode=True, max_batch_size=8):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        if precision_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        config.max_workspace_size = 16 * (1 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        for idx in range(len(shapes)):
            shapes[idx][0] = 1
            dynamic_shape = shapes[idx].copy()
            max_batch_shape = shapes[idx].copy()
            dynamic_shape[0] = -1
            max_batch_shape[0] = max_batch_size
            network.get_input(idx).shape = dynamic_shape
            profile.set_shape(
                network.get_input(idx).name, shapes[idx], shapes[idx],
                max_batch_shape)
        network.get_output(0).shape[0] = -1
        print(network.get_output(0).shape)
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
        return engine


def build_engine_trt7(onnx_path, shapes, precision_mode=True):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        #if builder.platform_has_fast_fp16 and precision_mode == 16:
        builder.fp16_mode = precision_mode
        builder.max_workspace_size = 16 * (1 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        for idx in range(len(shapes)):
            network.get_input(idx).shape = shapes[idx]
        engine = builder.build_cuda_engine(network)
        return engine


def build_engine(onnx_path, shapes, precision_mode=True, max_batch_size=8):
    if trt.__version__ > "8.0.0.0":
        return build_engine_trt8(onnx_path, shapes, precision_mode, max_batch_size)
    else:
        return build_engine_trt7(onnx_path, shapes, precision_mode)


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, plan_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def parser_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--onnx_model',
        type=str,
        default=os.path.join(work_root,
                             "onnx_models/checkpoint_epoch_00050.onnx"),
        help='test model file path',
    )
    parser.add_argument(
        '--trt_model',
        type=str,
        default=os.path.join(work_root, "trt_models/checkpoint_epoch_00050.engine"),
        help='tensorrt model file path',
    )
    return parser.parse_args()


def main():
    args = parser_args()
    print(args)
    onnx_path = args.onnx_model
    engine_name = args.trt_model

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())
    onnx_input = model.graph.input
    # print(onnx_input)
    input_shapes = []
    for i in range(len(onnx_input)):
        input_shape = []
        input_dim = len(onnx_input[i].type.tensor_type.shape.dim)
        for j in range(input_dim):
            dim = onnx_input[i].type.tensor_type.shape.dim[j].dim_value
            input_shape.append(dim)
        if len(input_shape):
            input_shapes.append(input_shape)
    print(input_shapes)  # [[1, 3, 4, 256, 256], [1, 3, 32, 256, 256]]

    engine = build_engine(onnx_path, input_shapes)
    save_engine(engine, engine_name)


if __name__ == '__main__':
    main()
    