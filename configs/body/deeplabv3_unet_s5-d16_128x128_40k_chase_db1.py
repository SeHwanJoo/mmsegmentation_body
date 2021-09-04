_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py',
    './dataset.py', '../_base_/default_runtime.py',
    './schedule_20k.py'
]
model = dict(
    decode_head=dict(
        num_classes=2,loss_decode=dict(
            _delete_=True, type='LovaszLoss', loss_weight=1.0, per_image=True)
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.25, 0.25, 0.5])
        ),
        # loss_decode=dict(
            # _delete_=True, type='FocalDiceLoss', loss_weight=1.0, focal_weight=0.75)
        # ),

    auxiliary_head=dict(
        num_classes=2,
        loss_decode=dict(
            _delete_=True, type='LovaszLoss', loss_weight=0.4, per_image=True)
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.25, 0.25, 0.5])
        ),
        # loss_decode=dict(
            # _delete_=True, type='FocalDiceLoss', loss_weight=0.4, focal_weight=0.75)
        # ),
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))
evaluation = dict(metric='mDice')

checkpoint_config = dict(max_keep_ckpts=5)