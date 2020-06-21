import os

import torch
from volksdep.converters import torch2trt, onnx2trt, load

from .base_task import BaseTask


class PyTorchClassifier(BaseTask):
    def __init__(self, model):
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            model.cuda()
        model.eval()
        super().__init__(model)

    def __call__(self, imgs):
        """
        Args:
            imgs (torch.float32): shape N*3*H*W

        Returns:
            feats (torch.float32): shape N*K, K is the number of classes
        """
        with torch.no_grad():
            if self.use_gpu:
                imgs = imgs.cuda()
            outp = self.model(imgs)
            if self.use_gpu:
                outp = outp.cpu()
            outp = outp.numpy()

        return outp


class TRTClassifier(BaseTask):
    def __init__(self, build_from, *args, **kwargs):
        if build_from == 'torch':
            func = torch2trt
        elif build_from == 'onnx':
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
            feats (np.float32): shape N*K, K is the number of classes
        """
        with torch.no_grad():
            imgs = imgs.cuda()
            outp = self.model(imgs)
            outp = outp.cpu()
        outp = outp.numpy()

        return outp


def build_classifier(source, *args, **kwargs):
    if source.lower() == 'pytorch':
        return PyTorchClassifier(*args, **kwargs)
    elif source.lower() == 'tensorrt':
        return TRTClassifier(*args, **kwargs)
    else:
        raise NotImplementedError('Classifier powered by %s not implemented' % source)
