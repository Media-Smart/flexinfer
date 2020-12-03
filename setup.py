from setuptools import dist
dist.Distribution().fetch_build_eggs(['cython'])

from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        raise EnvironmentError('CUDA is required to compile vedadet')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='flexinfer',
        version='2.0.0',
        description='Python Inference SDK',
        url='https://github.com/Media-Smart/flexinfer',
        author='Yichao Xiong, Hongxiang Cai',
        author_email='xyc_sjtu@163.com, chxlll@126.com',
        classifiers=[
            'Programming Language :: Python :: 3',
            'Operating System :: Linux',
            'License :: OSI Approved :: Apache Software License',
        ],
        keywords='computer vision, inference',
        packages=find_packages(),
        package_data={'flexinfer.ops': ['*/*.so']},
        setup_requires=[
            'tensorrt',
            'torch',
            'volksdep',
        ],
        install_requires=[
            'addict',
            'yapf',
            'numpy',
            'opencv-python',
        ],
        ext_modules=[
            make_cuda_ext(
                name='nms_ext',
                module='flexinfer.ops.nms',
                sources=['src/nms_ext.cpp', 'src/cpu/nms_cpu.cpp'],
                sources_cuda=[
                    'src/cuda/nms_cuda.cpp', 'src/cuda/nms_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
