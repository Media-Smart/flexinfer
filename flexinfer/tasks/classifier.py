import torch
from volksdep.converters import onnx2trt, load

from .base_task import BaseTask


class TRTClassifier(BaseTask):
    def __init__(self, checkpoint, *args, **kwargs):
        if checkpoint.endswith('onnx'):
            func = onnx2trt
            model = func(checkpoint, *args, **kwargs)
        elif checkpoint.endswith('engine'):
            func = load
            model = func(checkpoint)
        else:
            raise ValueError(
                'Unsupported build_from value %s, valid build_from value is torch, onnx and engine' % checkpoint)
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

        return outp


def build_classifier(*args, **kwargs):
    return TRTClassifier(*args, **kwargs)
