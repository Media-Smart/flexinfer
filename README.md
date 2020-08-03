# FlexInfer
A flexible Python front-end inference SDK.

## Features
- Flexible
  
  FlexInfer has a Python front-end, which makes it easy to build a computer vision product prototype.

- Efficient
  
  Most of time consuming part of FlexInfer is powered by C++ or CUDA, so FlexInfer is also efficient. If you are really hungry for efficiency and don't mind the trouble of C++, you can refer to [CheetahInfer](https://github.com/Media-Smart/cheetahinfer).

## License
This project is released under [Apache 2.0 license](https://github.com/Media-Smart/flexinfer/blob/master/LICENSE).

## Installation
### Requirements

- Linux
- Python 3.6 or higher
- TensorRT 7.1.3.4 or higher
- PyTorch 1.4.0 or higher
- CUDA 10.2 or higher
- volksdep

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- Python 3.6.9
- TensorRT 7.0.0.11
- PyTorch 1.4.0
- CUDA: 10.2

### Install FlexInfer

a. Install volksdep following the [official instructions](https://github.com/Media-Smart/volksdep)

b. If your platform is x86 or x64, you can create a conda virtual environment and activate it.

```shell
conda create -n flexinfer python=3.6.9 -y
conda activate flexinfer
```

c. Clone the flexinfer repository.

```shell
git clone https://github.com/Media-Smart/flexinfer
cd flexinfer
```

d. Install requirements.

```shell
pip install -r requirements.txt
```

## Usage
a. Generate onnx model or trt engine by using volksdep.

b. Example of deploying a classifier, you can run the following statement to classify an image.
```shell
python examples/classifier.py image_file checkpoint
```
c. Example of deploying a segmentor, you can run the following statement to generate segmentation mask.
```shell
python examples/segmentor.py image_file checkpoint
```
All sample files are in examples directory.

## Contact
This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).
