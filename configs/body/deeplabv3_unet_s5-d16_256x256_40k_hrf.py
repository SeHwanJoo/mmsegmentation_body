_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', './tumor_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(test_cfg=dict(crop_size=(512, 512), stride=(340, 340)))
evaluation = dict(metric='mDice')

checkpoint_config = dict(max_keep_ckpts=3)