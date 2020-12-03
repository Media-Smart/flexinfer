# 1. gpu id
gpu_id = 0

# 2. preprocess
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

preprocess = dict(
    typename='Compose',
    pipeline=[
        dict(typename='ToFloat', keys=['img']),
        dict(typename='PadIfNeeded', size=(513, 513),
             value=img_norm_cfg['mean'][::-1]),
        dict(typename='ImageToTensor', use_gpu=True),
        dict(typename='Normalize', **img_norm_cfg, use_gpu=True),
        dict(typename='Collect', keys=['img'])])

# 3. model
model = dict(
    typename='Onnx',
    model='unet_resnet101_voc.onnx',
    max_batch_size=1,
    fp16_mode=True)

# 4. postprocess
postprocess = dict(
    typename='Compose',
    pipeline=[
        dict(typename='Max', dim=1),
        dict(typename='InversePad', key='indices'),
        dict(typename='Collect', keys=['out'])
    ])
