_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', './dataset.py',
    '../_base_/default_runtime.py', './schedule_20k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=\
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth', # noqa
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        pretrain_style='official'),
     decode_head=dict(
        in_channels=768,
        in_index=3,
        channels=512,
        num_classes=3,
        loss_decode=dict(
            type='BCEDiceLoss', use_sigmoid=False, loss_weight=1.0, bce_weight=0.75)
        ),

    auxiliary_head=dict(
        in_channels=384,
        in_index=2,
        num_classes=3,
        channels=256,
        loss_decode=dict(
            type='BCEDiceLoss', use_sigmoid=False, loss_weight=0.4, bce_weight=0.75)
        ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(340, 340))
    )

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

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=400,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0)

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=1e-6,
    step=[30, 45])

evaluation = dict(metric='mDice')
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=5, norm_type=2))
checkpoint_config = dict(max_keep_ckpts=3)
# runner = dict(type='EpochBasedRunner', max_epochs=10)