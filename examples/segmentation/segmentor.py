import argparse
import sys
sys.path.insert(0, '../../')

import cv2
import numpy as np
from flexinfer.misc import Config, set_device
from flexinfer.preprocess import build_preprocess
from flexinfer.model import build_model
from flexinfer.postprocess import build_postprocess


PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def plot_result(pred_labels, image_file):
    image = cv2.imread(image_file)
    mask = np.zeros_like(image)

    for label, color in enumerate(PALETTE):
        mask[pred_labels == label, :] = color

    cover = (image * 0.5 + mask * 0.5).astype(np.uint8)

    cv2.imwrite('out.jpg', cover)


def segment(args):
    cfg = Config.fromfile(args.config)

    # 1. set gpu id
    set_device(cfg.gpu_id)

    # 2. build preprocess
    preprocess = build_preprocess(cfg.preprocess)

    # 3. build model
    model = build_model(cfg.model)

    # # 4. build postprocess
    postprocess = build_postprocess(cfg.postprocess)

    # 5. load image
    img = cv2.imread(args.image)
    data = [dict(img=img)]

    # 6. inference
    data = preprocess(data)
    data['out'] = model(data.pop('img'))
    pred_labels = postprocess(data)[0].cpu().numpy()

    # 7. plot result
    plot_result(pred_labels, args.image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation Inference Demo')
    parser.add_argument('config', help='config file')
    parser.add_argument('image', type=str, help='image file')

    args = parser.parse_args()
    segment(args)
