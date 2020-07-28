import torch
from volksdep.converters import onnx2trt, load

from .base_task import BaseTask


class TRTSegmentor(BaseTask):
    def __init__(self, build_from, *args, **kwargs):
        if build_from == 'onnx':
            func = onnx2trt
        elif build_from == 'engine':
            func = load
        else:
            raise ValueError('Unsupported build_from value %s, valid build_from value is torch, onnx and engine' % build_from)
        model = func(*args, **kwargs)
        super().__init__(model)

    def __call__(self, imgs):
        """
        Args:
            imgs (torch.float32): shape N*3*H*W

        Returns:
            feats (torch.): shape N*K*H*W, K is the number of classes
        """
        with torch.no_grad():
            imgs = imgs.cuda()
            outp = self.model(imgs)

        return outp


def build_segmentor(*args, **kwargs):
    return TRTSegmentor(*args, **kwargs)
