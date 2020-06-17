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
    def __init__(self, dst_shape):
        """
        Args:
            dst_shape(int list): [width, height]
        """
        self.dst_shape = dst_shape

    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image
        """
        if isinstance(img, np.ndarray):
            img = cv2.resize(img, self.dst_shape)
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return img


class Normalize:
    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        """
        Args:
            mean(float list): mean of [r, g, b] channel
            std(float list): mean of [r, g, b] channel
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image
        """
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
            img = (img - self.mean) / self.std
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return img


class ColorSpaceConvert:
    def __init__(self, convert_flag=cv2.COLOR_BGR2RGB):
        self.convert_flag = convert_flag

    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image
        """
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, self.convert_flag)
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return img


class ToTensor:
    def __call__(self, img):
        """
        Args:
            img(np.ndarray): image
        """
        if isinstance(img, np.ndarray):
            # handle numpy array
            if img.ndim == 2:
                img = img[:, :, None]
            img = img.astype(np.float32)
            img = torch.from_numpy(img.transpose((2, 0, 1)))
        else:
            raise TypeError('img shoud be np.ndarray. Got %s' % type(img))
        return img


class Batchify:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        imgs_tf = []
        for img in imgs:
            img_tf = self.transform(img)
            imgs_tf.append(img_tf)
        tensor = torch.stack(imgs_tf, dim=0)
        return tensor
