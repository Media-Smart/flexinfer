# 1. gpu id
gpu_id = 0

# 2. preprocess
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size_divisor = 32

preprocess = dict(
    typename='Compose',
    pipeline=[
        dict(typename='Resize', dst_shape=(800, 1333), keep_ratio=True),
        dict(typename='ToFloat', keys=['img']),
        dict(typename='PadIfNeeded', size_divisor=128,
             value=img_norm_cfg['mean'][::-1]),
        dict(typename='ImageToTensor', use_gpu=True),
        dict(typename='Normalize', **img_norm_cfg, use_gpu=True),
        dict(typename='Collect', keys=['img'])])

# 3. model
model = dict(
    typename='Onnx',
    model='retinanet_r50.onnx',
    max_batch_size=1,
    min_input_shapes=[(3, 128, 128)],       # Should be set when onnx model has dynamic shapes, the shape format is CxHxW. Otherwise, set None.
    max_input_shapes=[(3, 1408, 1408)],     # Should be set when onnx model has dynamic shapes, the shape format is CxHxW. Otherwise, set None.
    fp16_mode=True)

# 4. postprocess
num_classes = 80
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [0.5, 1.0, 2.0]
num_anchors = scales_per_octave * len(ratios)

meshgrid = dict(
    typename='BBoxAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='BBoxBaseAnchor',
        octave_base_scale=4,
        scales_per_octave=scales_per_octave,
        ratios=ratios,
        base_sizes=strides))

bbox_coder = dict(
    typename='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0])

converter = dict(
    typename='BBoxAnchorConverter',
    num_classes=num_classes,
    bbox_coder=bbox_coder,
    nms_pre=1000,
    use_sigmoid=use_sigmoid)

infer_cfg = dict(
    min_bbox_size=0,
    score_thr=0.6,
    nms=dict(
        typename='nms',
        iou_thr=0.5),
    max_per_img=100)

postprocess = dict(
    typename='Compose',
    pipeline=[
        dict(typename='ObjDetPostProcess', meshgrid=meshgrid,
             converter=converter, num_classes=num_classes,
             use_sigmoid=use_sigmoid, infer_cfg=infer_cfg),
        dict(typename='Collect', keys=['out'])
    ])

# 5. class name
class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
