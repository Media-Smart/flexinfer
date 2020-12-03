from flexinfer.misc import registry


@registry.register_module('postprocess')
class Collect:
    def __init__(self, keys=['out']):
        self.keys = keys

    def __call__(self, results):
        out = [results[key] for key in self.keys]
        if len(out) == 1:
            out = out[0]

        return out
