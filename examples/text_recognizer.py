import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))

from flexinfer.inference import build_inferencer
from flexinfer.preprocess import preprocess as TF
from flexinfer.postprocess import SoftmaxProcess, IndexToString, Compose
from flexinfer.utils import set_device


def main(imgfp):
    gpu_id = 0
    # 1. set gpu id, default gpu id is 0
    set_device(gpu_id=gpu_id)
    use_gpu = torch.cuda.is_available()

    # 2. prepare for transfoms and model
    ## 2.1 transforms
    transform = TF.Compose([
        TF.Resize(dst_shape=(100, 32), interpolation=cv2.INTER_LINEAR),
        TF.ToTensor(use_gpu=use_gpu),
        TF.Normalize(mean=127.5, std=127.5, use_gpu=use_gpu, gray=True),
    ])
    batchify = TF.Batchify(transform)

    ## 2.2 post process
    postprocess = Compose([
        SoftmaxProcess(dim=2),
        IndexToString(character='abcdefghijklmnopqrstuvwxyz0123456789'),
    ])

    ## 2.3 model
    ## build text recognizer with trt engine from onnx model or serialized engine
    text_recognizer = build_inferencer(checkpoint=args.checkpoint, max_batch_size=2,
                                       fp16_mode=True)

    # 3. load image
    img = cv2.imread(imgfp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

    # 4. inference
    imgs = [img, img]
    tensor = batchify(imgs)
    outp = text_recognizer(tensor)
    outp = postprocess(outp)
    print(outp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TextReocgnizer demo')
    parser.add_argument('checkpoint',
                        type=str, help='checkpoint file path')
    parser.add_argument('imgfp', type=str, help='path to image file path')
    args = parser.parse_args()
    main(args.imgfp)
