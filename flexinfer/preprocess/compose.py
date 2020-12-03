import torch
from flexinfer.misc import registry, build_from_cfg


@registry.register_module('preprocess')
class Compose:
    """Composes several operations together.
    Args:
        preprocess (list): list of operations to compose.
    """

    def __init__(self, pipeline):
        self.pipeline = [build_from_cfg(cfg, registry, 'preprocess') for cfg
                         in pipeline]

    def _process(self, results):
        assert isinstance(results, dict)

        for transform in self.pipeline:
            results = transform(results)

        return results

    def __call__(self, results):
        assert isinstance(results, list)

        results = list(map(self._process, results))

        outputs = dict()
        for key in results[0]:
            outputs[key] = [res[key] for res in results]

        outputs['img'] = torch.cat(outputs['img'], dim=0)

        return outputs
