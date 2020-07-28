import os
import sys
import argparse

import cv2
import torch

sys.path.insert(0, os.path.abspath('.'))

from flexinfer.tasks import build_classifier
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
    ### build classifier with trt engine from onnx model
    classifier = build_classifier(build_from='onnx', model='resnet18.onnx',
                                  max_batch_size=2, fp16_mode=True)
    ### build classifier with trt engine from serialized engine
    # classifier = build_classifier(
    #                             build_from='engine', engine='resnet18.engine')

    # 3. load image
    img = cv2.imread(imgfp)

    imgs = [img, img]

    tensor = batchify(imgs)
    outp = classifier(tensor)
    print(outp.argmax(1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier demo')
    parser.add_argument('imgfp', type=str, help='path to image file path')
    args = parser.parse_args()
    main(args.imgfp)
