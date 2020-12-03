from abc import ABCMeta, abstractmethod

import torch
from flexinfer.misc import registry


class Base(metaclass=ABCMeta):
    def __init__(self, key='out', out='out'):
        self.key = key
        self.out = out

    @abstractmethod
    def postprocess(self, results):
        pass

    def __call__(self, results):
        self.postprocess(results)

        return results


@registry.register_module('postprocess')
class Softmax(Base):
    def __init__(self, dim=-1, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.dim = dim

    def postprocess(self, results):
        results[self.out] = results[self.key].softmax(dim=self.dim)


@registry.register_module('postprocess')
class Sigmoid(Base):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def postprocess(self, results):
        results[self.out] = results[self.key].sigmoid()


@registry.register_module('postprocess')
class Max(Base):
    def __init__(self, dim=-1, out=['values', 'indices'], **kwargs):
        super(Max, self).__init__(out=out, **kwargs)
        self.dim = dim

    def postprocess(self, results):
        for name, res in zip(self.out, results[self.key].max(dim=self.dim)):
            results[name] = res


@registry.register_module('postprocess')
class Mask(Base):
    def __init__(self, threshold=0.5, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.threshold = threshold

    def postprocess(self, results):
        results[self.out] = (results[self.key] >= self.threshold).to(torch.int)


@registry.register_module('postprocess')
class InversePad(Base):
    def __init__(self, **kwargs):
        super(InversePad, self).__init__(**kwargs)

    def postprocess(self, results):
        out = []
        for data, meta_data in zip(results[self.key], results['img_metas']):
            h, w = meta_data['img_shape'][:2]
            data = data[..., :h, :w]

            out.append(data)

        results[self.out] = out
