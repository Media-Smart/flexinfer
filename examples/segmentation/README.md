## Example

Here is an example for segmentation.

1. Download the onnx model and test image from 
[here](https://drive.google.com/drive/folders/1-Bv5cYFkX4AreO-8vp1M5Of64iavI448?usp=sharing), 
and put them in the same directory as segmentor.py and unet_resnet101_voc.py.

2. Run
```shell
python segmentor.py unet_resnet101_voc.py test.jpg
```