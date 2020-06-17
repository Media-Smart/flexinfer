import sys
import os
import argparse

import torch
from torchvision.models import resnet18
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath('.'))

from flexinfer.tasks import build_classifier
from flexinfer.preprocess import transforms as TF


def main(imgfp):
    # 1. prepare for transfoms and model
    ## 1.1 transforms
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ColorSpaceConvert(),
        TF.Normalize(),
        TF.ToTensor(),
    ])
    batchify = TF.Batchify(transform)

    ## 1.2 model
    model = resnet18(pretrained=True)
    classifier = build_classifier('pytorch', model, gpu_id=1)

    # 2. load image
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
