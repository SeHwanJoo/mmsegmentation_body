_base_ = [
    '../_base_/models/upernet_swin_BN.py', 'dataset1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained=\
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth', # noqa
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        pretrain_style='official'),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=3),
    auxiliary_head=dict(in_channels=512, num_classes=3)
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

lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=1,
    step_ratio_up=0.05)

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=400,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0)

# lr_config = dict(
#     _delete_=True,
#     policy='step',
#     warmup='linear',
#     warmup_iters=400,
#     warmup_ratio=1e-6,
#     step=[60, 90])

evaluation = dict(metric='mDice')
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(max_keep_ckpts=3)