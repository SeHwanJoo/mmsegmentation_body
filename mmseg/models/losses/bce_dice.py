# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss import cross_entropy

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
from .dice_loss import dice_loss
from .cross_entropy_loss import  (cross_entropy, binary_cross_entropy, 
                                  mask_cross_entropy)



@LOSSES.register_module()
class BCEDiceLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 bce_weight = 0.5,
                 **kwards):
        super(BCEDiceLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.bce_weight = bce_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_bce = self.loss_weight * self.cls_criterion(
            pred,
            target,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwards)
        
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss_dice = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        
        
        return (loss_bce * self.bce_weight) + (loss_dice * (1-self.bce_weight))
