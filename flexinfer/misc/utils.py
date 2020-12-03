import os
import os.path as osp


def set_device(gpu_id=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
