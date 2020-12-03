from flexinfer.misc import registry, build_from_cfg


def build_model(cfg):
    return build_from_cfg(cfg, registry, 'inference')
