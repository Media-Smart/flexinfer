import os

import torch
from volksdep.converters import TRTEngine

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
        if self.use_gpu:
            imgs = imgs.cuda()
        with torch.no_grad():
            outp = self.model(imgs)
        if self.use_gpu:
            outp = outp.cpu()
        outp = outp.numpy()

        return outp


class TRTClassifier(BaseTask):
    def __init__(self, build_from, *args, **kwargs):
        model = TRTEngine(build_from, *args, **kwargs)
        super().__init__(model)

    def __call__(self, imgs):
        """
        Args:
            imgs (torch.float32): shape N*3*H*W

        Returns:
            feats (np.float32): shape N*K, K is the number of classes
        """
        outp = self.model.inference(imgs)

        return outp


def build_classifier(source, *args, **kwargs):
    if source.lower() == 'pytorch':
        return PyTorchClassifier(*args, **kwargs)
    elif source.lower() == 'tensorrt':
        return TRTClassifier(*args, **kwargs)
    else:
        raise NotImplementedError('Classifier powered by %s not implemented' % source)
