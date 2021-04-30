import argparse
from enum import Enum

import cv2
import numpy as np

from flexinfer.misc import Config, set_device
from flexinfer.preprocess import build_preprocess
from flexinfer.model import build_model
from flexinfer.postprocess import build_postprocess


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


class Color(Enum):
    """An enum that defines common colors.
    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.
    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs
    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = cv2.imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    cv2.imwrite(outfp, img)


def detect(args):
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
    result = postprocess(data)[0]

    # 7. plot result
    plot_result(result, args.image, cfg.class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Inference Demo')
    parser.add_argument('config', help='config file')
    parser.add_argument('image', type=str, help='image file')

    args = parser.parse_args()
    detect(args)
