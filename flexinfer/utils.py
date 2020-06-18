import os


def set_device(gpu_id=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
