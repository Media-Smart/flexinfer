from flexinfer.misc import registry, build_from_cfg


def build_postprocess(cfg):
    return build_from_cfg(cfg, registry, 'postprocess')
