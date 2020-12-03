import torch
import numpy as np
from flexinfer.misc import registry


DEFAULT = dict(
    scale_factor=1.0,
    flip=False,
    flip_direction=None,
)


@registry.register_module('preprocess')
class ToFloat:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].astype(np.float32)

        return results


@registry.register_module('preprocess')
class ImageToTensor:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def __call__(self, results):
        """
        Args:
            img(torch.Tensor): image, shape 1*C*H*W
        """

        img = results['img']
        if img.ndim == 2:
            img = img[:, :, None]
        img = torch.from_numpy(img)
        if self.use_gpu:
            img = img.cuda()
        img = img.permute(2, 0, 1)  # h*w*c -> c*h*w
        img = img.float()
        img = img.unsqueeze(0)  # c*h*w -> 1*c*h*w

        results['img'] = img

        return results


@registry.register_module('preprocess')
class Collect:
    def __init__(self,
                 keys,
                 meta_keys=['img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction']):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}

        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            else:
                img_meta[key] = DEFAULT[key]

        for key in self.keys:
            data[key] = results[key]
        data['img_metas'] = img_meta

        return data
