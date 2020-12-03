import cv2
import numpy as np
import torch
from flexinfer.misc import registry


@registry.register_module('preprocess')
class ImageToGray:
    def __call__(self, results):
        img = results['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        results['img'] = img

        return results


@registry.register_module('preprocess')
class Resize:
    def __init__(self, dst_shape, interpolation=cv2.INTER_LINEAR,
                 keep_ratio=False):
        """
        Args:
            dst_shape(int list): [height, width]
        """

        self.dst_shape = dst_shape
        self.interp = interpolation
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        """
        Args:
            img(np.ndarray): image, shape H*W*C
        """

        img = results['img']
        h, w = img.shape[:2]
        new_h, new_w = self.dst_shape
        if self.keep_ratio:
            scale_factor = min(max(new_h, new_w) / max(h, w),
                               min(new_h, new_w) / min(h, w))
            new_h = int(h * float(scale_factor) + 0.5)
            new_w = int(w * float(scale_factor) + 0.5)

        img = cv2.resize(img, (new_w, new_h), interpolation=self.interp)  # TODO time consuming
        w_scale = new_w / w
        h_scale = new_h / h

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor

        return results


@registry.register_module('preprocess')
class PadIfNeeded:
    def __init__(self, size=None, size_divisor=None, mode=cv2.BORDER_CONSTANT,
                 value=(123.675, 116.280, 103.530)):
        self.size = size
        self.size_divisor = size_divisor
        self.mode = mode
        self.value = np.array(value)
        self.size_divisor = size_divisor

        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self, results):
        """
        Args:
            img(np.ndarray): image, shape H*W*C
        """

        img = results['img']
        h, w = img.shape[:2]

        if self.size is not None:
            pad_h, pad_w = self.size
            assert h <= pad_h and w <= pad_w
        else:
            pad_h = int(np.ceil(h / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(w / self.size_divisor)) * self.size_divisor
        padded_img = cv2.copyMakeBorder(img, 0, pad_h - h, 0, pad_w - w,
                                        self.mode, value=self.value)

        results['img'] = padded_img
        results['img_shape'] = img.shape
        results['pad_shape'] = padded_img.shape

        return results


@registry.register_module('preprocess')
class Normalize:
    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 use_gpu=True,
                 gray=False,
                 to_rgb=True):
        """
        Args:
            mean(float list): mean of [r, g, b] channel
            std(float list): std of [r, g, b] channel
        """
        # n*c*h*w

        shape = (1, 1, 1, 1) if gray else (1, 3, 1, 1)

        self.mean = torch.tensor(mean, dtype=torch.float32).view(*shape)
        self.std = torch.tensor(std, dtype=torch.float32).view(*shape)
        self.gray = gray
        self.to_rgb = to_rgb
        if use_gpu:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def __call__(self, results):
        """
        Args:
            img(torch.Tensor): image, shape 1*C*H*W
        """

        img = results['img']
        if not self.gray and self.to_rgb:
            # n*c*h*w
            img = img[:, [2, 1, 0], :, :]  # bgr to rgb
        img = (img - self.mean) / self.std  # time consuming on cpu

        results['img'] = img

        return results
