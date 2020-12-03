from flexinfer.misc import build_from_cfg, registry, singleton_arg


@singleton_arg
def build_preprocess(cfg):
    return build_from_cfg(cfg, registry, 'preprocess')
