import cv2
import torch
import torch.nn.functional as NF
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
    def __init__(self, dst_shape):
        """
        Args:
            dst_shape(int list): [width, height]
        """
        self.dst_shape = dst_shape

    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image, shape 1*C*H*W
        """
        if isinstance(img, np.ndarray):
            rimg = cv2.resize(img, self.dst_shape)  # TODO time consuming
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return rimg


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
                 use_gpu=True,
                 bgr2rgb=True):
        """
        Args:
            mean(float list): mean of [r, g, b] channel
            std(float list): std of [r, g, b] channel
        """
        # n*c*h*w
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.bgr2rgb = bgr2rgb
        if use_gpu is not None:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def __call__(self, img):
        """
        Args:
            img(torch.Tensor): image, shape 1*C*H*W
        """
        if isinstance(img, torch.Tensor):
            if self.bgr2rgb:
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
