_base_ = [
    '../_base_/models/upernet_swin_BN.py', 'dataset2.py',
    '../_base_/default_runtime.py', './schedule_20k.py'
]
model = dict(
    pretrained=\
        'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',  # noqa
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        pretrain_style='official'),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=3,
        loss_decode=dict(
            _delete_=True, type='LovaszLoss', loss_weight=1.0, per_image=True)
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.25, 0.25, 0.5])
        ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=3,
        loss_decode=dict(
            _delete_=True, type='LovaszLoss', loss_weight=0.4, per_image=True)
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.25, 0.25, 0.5])
        )
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# lr_config = dict(
#     _delete_=True,
#     policy='cyclic',
#     target_ratio=(1, 0.01),
#     cyclic_times=1,
#     step_ratio_up=0.05)

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=1e-6,
    step=[30, 45])


# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8)
checkpoint_config = dict(max_keep_ckpts=5)
