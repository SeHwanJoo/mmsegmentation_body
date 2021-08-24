_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py', './tumor_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', eps=0.001, requires_grad=True)
model = dict(
    pretrained='open-mmlab://contrib/mobilenet_v3_large',
    type='EncoderDecoder',
    backbone=dict(
        type='MobileNetV3',
        arch='small',
        out_indices=(0, 1, 12),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 16, 576),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=3,
        )
    )
# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)

runner = dict(type='IterBasedRunner', max_iters=320000)
