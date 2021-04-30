## Example

Here are examples for object detection.

1. Download the onnx model and test image from [RetinaNet](https://drive.google.com/drive/folders/1-fkcTOxjsSLX1k1KO7DzK1QGYGxiz9JP?usp=sharing) or [TinaFace](https://drive.google.com/drive/folders/1kyh7yu1LmfPwigseA4XvgdMqTCRhmATY?usp=sharing), and put them in the same directory as detector.py.

2. Run
```shell
# RetinaNet
python detector.py retinanet.py test_retinanet.jpg

# TinaFace
python detector.py tinaface.py test_tinaface.jpg
```
