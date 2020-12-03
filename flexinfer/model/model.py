import torch
from volksdep.converters import onnx2trt, load
from flexinfer.misc import registry


class Base:
    def __init__(self, model):
        self.model = model

    def __call__(self, imgs):
        """
        Args:
            imgs (torch.float32): shape N*C*H*W

        Returns:
            outp (torch.float32)
        """
        imgs = imgs.cuda()
        outp = self.model(imgs)

        return outp


@registry.register_module('inference')
class Onnx(Base):
    def __init__(self, *args, **kwargs):

        """build TensorRT model from Onnx model.

        Args:
            model (string or io object): Onnx model name
            log_level (string, default is ERROR): TensorRT logger level, now
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
            max_batch_size (int, default=1): The maximum batch size which can be
                used at execution time, and also the batch size for which the
                ICudaEngine will be optimized.
            max_workspace_size (int, default is 1): The maximum GPU temporary
                memory which the ICudaEngine can use at execution time. default is
                1GB.
            fp16_mode (bool, default is False): Whether or not 16-bit kernels are
                permitted. During engine build fp16 kernels will also be tried when
                this mode is enabled.
            strict_type_constraints (bool, default is False): When strict type
                constraints is set, TensorRT will choose the type constraints that
                conforms to type constraints. If the flag is not enabled higher
                precision implementation may be chosen if it results in higher
                performance.
            int8_mode (bool, default is False): Whether Int8 mode is used.
            int8_calibrator (volksdep.calibrators.base.BaseCalibrator,
                default is None): calibrator for int8 mode, if None, default
                calibrator will be used as calibration data.
        """

        model = onnx2trt(*args, **kwargs)

        super(Onnx, self).__init__(model)


@registry.register_module('inference')
class TRTEngine(Base):
    def __init__(self, *args, **kwargs):
        """build trt engine from saved engine
        Args:
            engine (string): engine file name to load
            log_level (string, default is ERROR): tensorrt logger level,
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        """

        model = load(*args, **kwargs)
        super(TRTEngine, self).__init__(model)
