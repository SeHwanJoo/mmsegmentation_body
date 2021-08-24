# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TumorDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('background', 'kidney', 'tumor')

    PALETTE = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]

    def __init__(self, **kwargs):
        super(TumorDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

