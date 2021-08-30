# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/swin_ohem2.py --deterministic --seed 28
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/swin1.py --deterministic --seed 28
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/swin_ohem1.py --deterministic --seed 28
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/swin2.py --deterministic --seed 28
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/upernet_swin_tiny.py --deterministic --seed 28
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/upernet_swin_tiny1.py --deterministic --seed 28
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/upernet_swin_tiny2.py --deterministic --seed 28
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/body/upernet_swin_tiny3.py --deterministic --seed 28
