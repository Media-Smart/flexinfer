import torch
from volksdep.converters import onnx2trt, load


class Inferencer:
    def __init__(self, checkpoint, *args, **kwargs):
        if checkpoint.endswith('onnx'):
            func = onnx2trt
            self.model = func(checkpoint, *args, **kwargs)
        elif checkpoint.endswith('engine'):
            func = load
            self.model = func(checkpoint)
        else:
            raise ValueError(
                'Unsupported build_from value %s, valid build_from value is torch, onnx and engine' % checkpoint)

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


def build_inferencer(*args, **kwargs):
    return Inferencer(*args, **kwargs)
