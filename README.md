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

b. Example of deploying a task (eg. classifier), you can run the following statement:
```shell
python examples/classifier.py checkpoint_path image_file
```

## [Media-smart toolboxes](https://github.com/Media-Smart)

You can use the following toolboxes to generate your model.

a. [vedacls](https://github.com/Media-Smart/vedacls) an open source classification toolbox based on PyTorch.

b. [vedaseg](https://github.com/Media-Smart/vedaseg) an open source semantic segmentation toolbox based on PyTorch.

c. [vedastr](https://github.com/Media-Smart/vedastr) open source scene text recognition toolbox based on PyTorch.

## Contact
This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), 
Yichao Xiong ([@mileistone](https://github.com/mileistone)) Yuxin Zou ([@Yuxin Zou](https://github.com/YuxinZou))
and Jun Sun([@ChaseMonsterAway](https://github.com/ChaseMonsterAway)).
