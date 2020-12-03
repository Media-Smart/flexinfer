from flexinfer.misc import registry, build_from_cfg


@registry.register_module('postprocess')
class Compose:
    """Composes several operations together.
    Args:
        postprocess (list): list of operations to compose.
    """

    def __init__(self, pipeline):
        self.pipeline = [build_from_cfg(cfg, registry, 'postprocess') for cfg
                         in pipeline]

    def __call__(self, results):
        for postprocess in self.pipeline:
            results = postprocess(results)

        return results
