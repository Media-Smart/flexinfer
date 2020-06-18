from abc import ABCMeta, abstractmethod


class BaseTask(metaclass=ABCMeta):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def __call__(self, imgs):
        pass
