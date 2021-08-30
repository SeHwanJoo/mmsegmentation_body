_base_ = [
    '../_base_/models/upernet_r50_BN.py', './dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='open-mmlab://resnext50_32x4d',
    backbone=dict(
        _delete_=True,
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    decode_head=dict(num_classes=3), 
    auxiliary_head=dict(num_classes=3)
    )

checkpoint_config = dict(max_keep_ckpts=1)