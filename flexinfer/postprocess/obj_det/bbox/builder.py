from flexinfer.misc import build_from_cfg, registry, singleton_arg


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, registry, 'bbox_assigner', default_args)


@singleton_arg
def build_bbox_coder(cfg, **default_args):
    bbox_coder = build_from_cfg(cfg, registry, 'bbox_coder', default_args)
    return bbox_coder
