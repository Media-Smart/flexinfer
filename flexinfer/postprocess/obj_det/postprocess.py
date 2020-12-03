from flexinfer.misc import registry

from .meshgrids import build_meshgrid
from .converters import build_converter
from .bbox import multiclass_nms, bbox2result
from ..base import Base


@registry.register_module('postprocess')
class ObjDetPostProcess(Base):
    def __init__(self, meshgrid, converter, num_classes, use_sigmoid,
                 infer_cfg, **kwargs):
        super(ObjDetPostProcess, self).__init__(**kwargs)

        self.meshgrid = build_meshgrid(meshgrid)
        self.converter = build_converter(converter)
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.infer_cfg = infer_cfg

    def _get_raw_dets(self, feats, img_metas):
        feats_len = len(feats)
        assert feats_len % 2 == 0
        half_len = feats_len // 2
        feats = [feats[:half_len], feats[half_len:]]

        featmap_sizes = [feat.shape[-2:] for feat in feats[0]]
        dtype = feats[0][0].dtype
        device = feats[0][0].device
        anchor_mesh = self.meshgrid.gen_anchor_mesh(featmap_sizes, img_metas,
                                                    dtype, device)
        # bboxes, scores, score_factor
        dets = self.converter.get_bboxes(anchor_mesh, img_metas, *feats)

        return dets

    def postprocess(self, results):
        dets = self._get_raw_dets(results[self.key], results['img_metas'])
        batch_size = len(dets)

        result_list = []
        for ii in range(batch_size):
            bboxes, scores, centerness = dets[ii]
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                self.infer_cfg.score_thr,
                self.infer_cfg.nms,
                self.infer_cfg.max_per_img,
                score_factors=centerness)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.cls_out_channels)
            result_list.append(bbox_result)

        results[self.out] = result_list
