from abc import ABCMeta, abstractmethod


class BaseTask(metaclass=ABCMeta):
    def __init__(self, model, gpu_id=None):
        if gpu_id is not None:
            model = model.cuda(gpu_id)
        self.model = model
        self.gpu_id = gpu_id

    @abstractmethod
    def __call__(self, imgs):
        pass
