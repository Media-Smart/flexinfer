import cv2
import torch
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image
        """
        for tf in self.transforms:
            img = tf(img)
        return img


class Resize:
    def __init__(self, dst_shape, interp):
        """
        Args:
            dst_shape(int list): [width, height]
        """
        self.dst_shape = dst_shape
        self.interp = interp

    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image, shape H*W*C
        """
        if isinstance(img, np.ndarray):
            rimg = cv2.resize(img, self.dst_shape, interpolation=self.interp)  # TODO time consuming
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return rimg


class PadIfNeeded:
    def __init__(self, height, width, mode=cv2.BORDER_CONSTANT,
                 value=(123.675, 116.280, 103.530)):
        self.height = height
        self.width = width
        self.mode = mode
        self.value = value

    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image, shape H*W*C
        """
        h, w, c = img.shape

        assert h <= self.height and w <= self.width
        if isinstance(img, np.ndarray):
            img = cv2.copyMakeBorder(img, 0, self.height - h, 0, self.width - w,
                                     self.mode, value=self.value)
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return img


class ToTensor:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def __call__(self, img):
        """
        Args:
            img(torch.Tensor): image, shape 1*C*H*W
        """
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            img = torch.from_numpy(img)
            if self.use_gpu:
                img = img.cuda()
            img = img.permute(2, 0, 1)  # h*w*c -> c*h*w
            img = img.float()
            img = img.unsqueeze(0)  # c*h*w -> 1*c*h*w
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return img


class Normalize:
    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 gray=False,
                 use_gpu=True,
                 bgr2rgb=True):
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
        self.bgr2rgb = bgr2rgb
        if use_gpu:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def __call__(self, img):
        """
        Args:
            img(torch.Tensor): image, shape 1*C*H*W
        """
        if isinstance(img, torch.Tensor):
            if self.bgr2rgb and not self.gray:
                # n*c*h*w
                img = img[:, [2, 1, 0], :, :]  # bgr to rgb
            img = (img - self.mean) / self.std  # time consuming on cpu
        else:
            raise TypeError('img shoud be torch.Tensor. Got %s' % type(img))
        return img


class Batchify:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        imgs_tf = []
        for img in imgs:
            img_tf = self.transform(img)
            imgs_tf.append(img_tf)
        imgs_tf = torch.cat(imgs_tf, dim=0)  # time consuming on cpu
        return imgs_tf
