import torch

from ..preprocess import transforms as TF
from .base_task import BaseTask


class PyTorchClassifier(BaseTask):
    def __init__(self, model, gpu_id=None):
        model.eval()
        super().__init__(model, gpu_id)

    def __call__(self, imgs):
        """
        Args:
            imgs (torch.float32): shape N*3*H*W

        Returns:
            feats (torch.float32): shape N*K, K is the number of classes
        """
        if self.gpu_id is not None:
            imgs = imgs.cuda(self.gpu_id)
        with torch.no_grad():
            outp = self.model(imgs)
        if self.gpu_id is not None:
            outp = outp.cpu()
        outp = outp.numpy()
        return outp


def build_classifier(source, *args, **kwargs):
    if source.lower() == 'pytorch':
        return PyTorchClassifier(*args, **kwargs)
    else:
        raise NotImplementedError('Classifier powered by %s not implemented' % source)
