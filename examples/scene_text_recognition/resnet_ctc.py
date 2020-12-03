# 1. gpu id
gpu_id = 0

# 2. preprocess
img_norm_cfg = dict(mean=127.5, std=127.5, gray=True)

preprocess = dict(
    typename='Compose',
    pipeline=[
        dict(typename='ImageToGray'),
        dict(typename='Resize', dst_shape=(32, 100)),
        dict(typename='ImageToTensor', use_gpu=True),
        dict(typename='Normalize', **img_norm_cfg, use_gpu=True),
        dict(typename='Collect', keys=['img'])
    ])

# 3. model
model = dict(
    typename='Onnx',
    model='resnet_ctc.onnx',
    max_batch_size=2,
    fp16_mode=True)

# 4. postprocess
postprocess = dict(
    typename='Compose',
    pipeline=[
        dict(typename='Max', dim=-1),
        dict(typename='IndexToString',
             key='indices',
             character='abcdefghijklmnopqrstuvwxyz0123456789'),
        dict(typename='Collect', keys=['out'])
    ])
