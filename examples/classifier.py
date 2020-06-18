import os
import sys
import argparse

import torch
from torchvision.models import resnet18
import cv2

sys.path.insert(0, os.path.abspath('.'))

from flexinfer.tasks import build_classifier
from flexinfer.preprocess import transforms as TF
from flexinfer.utils import set_device


def main(imgfp):
    # 1. set gpu id, default gpu id is 0
    set_device(gpu_id=0)

    # 2. prepare for transfoms and model
    ## 2.1 transforms
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ColorSpaceConvert(),
        TF.Normalize(),
        TF.ToTensor(),
    ])
    batchify = TF.Batchify(transform)

    ## 2.2 model
    model = resnet18(pretrained=True)
    ### build classifier with pytorch model
    classifier = build_classifier('pytorch', model)
    ### build classifier with trt engine from pytorch model
    # classifier = build_classifier('tensorrt', build_from='torch', model=model, dummy_input=torch.randn(1, 3, 224, 224), fp16_mode=True)
    ### build classifier with trt engine from onnx model
    # classifier = build_classifier('tensorrt', build_from='onnx', model='resnet18.onnx', fp16_mode=True)
    ### build classifier with trt engine from serialized engine
    # classifier = build_classifier('tensorrt', build_from='engine', model='resnet18.engine')

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
