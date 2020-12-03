import argparse

import cv2
from flexinfer.misc import Config, set_device
from flexinfer.preprocess import build_preprocess
from flexinfer.model import build_model
from flexinfer.postprocess import build_postprocess


def recognize(args):
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
    data = [dict(img=img), dict(img=img)]   # suppose there are two images

    # 6. inference
    data = preprocess(data)
    data['out'] = model(data.pop('img'))
    res = postprocess(data)

    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification Inference Demo')
    parser.add_argument('config', help='config file')
    parser.add_argument('image', type=str, help='image file')

    args = parser.parse_args()
    recognize(args)
