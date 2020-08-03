import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.abspath('.'))

from flexinfer.tasks import build_inferencer
from flexinfer.preprocess import transforms as TF
from flexinfer.utils import set_device


def main(imgfp):
    gpu_id = 0
    # 1. set gpu id, default gpu id is 0
    set_device(gpu_id=gpu_id)
    use_gpu = torch.cuda.is_available()

    # 2. prepare for transfoms and model
    ## 2.1 transforms
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ToTensor(use_gpu=use_gpu),
        TF.Normalize(use_gpu=use_gpu),
    ])
    batchify = TF.Batchify(transform)

    ## 2.2 model
    ###build segmentor with trt engine from onnx model or serialized engine
    classifier = build_inferencer(checkpoint=args.checkpoint,
                                  max_batch_size=2, fp16_mode=True)

    # 3. load image
    img = cv2.imread(imgfp)

    imgs = [img, img]

    tensor = batchify(imgs)
    outp = classifier(tensor)
    print(outp.argmax(1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier demo')
    parser.add_argument('checkpoint',
                        type=str, help='checkpoint file path')
    parser.add_argument('imgfp', type=str, help='path to image file path')
    args = parser.parse_args()
    main(args.imgfp)
