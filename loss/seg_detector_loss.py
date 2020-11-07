import torch
import torch.nn as nn

"""
Build loss functions for SegDetector.
Details about the built functions:
Input:
    pred: A dict which contains predictions.
        prob: The text segmentation prediction.
        thresh: The threshold prediction        
        binary: Value produced by `step_function(prob - thresh)`.
    target:
        gt: Text regions bitmap gt.
        mask: Ignore mask, pixels where value is 1 indicates no contribution to loss.
        thresh_mask: Mask indicates regions cared by thresh supervision.
        thresh_map: Threshold gt.
Return:
    (loss, metrics).
    loss: A scalar loss value.
    metrics: A dict contraining partial loss values.
"""


class DiceLoss(nn.Module):
    '''
    DiceLoss on prob.
    For SegDetector without adaptive module.
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        from .dice_loss import DiceLoss as Loss
        self.loss = Loss(eps)

    def forward(self, pred, target):
        loss = self.loss(pred['prob'], target['gt'], target['mask'])
        return loss, dict(dice_loss=loss)


class BalanceBCELoss(nn.Module):
    '''
    DiceLoss on prob.
    For SegDetector without adaptive module.
    '''

    def __init__(self, eps=1e-6):
        super(BalanceBCELoss, self).__init__()
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        self.loss = BalanceCrossEntropyLoss(eps=eps)

    def forward(self, pred, target):
        loss = self.loss(pred['prob'], target['gt'], target['mask'])
        return loss, dict(dice_loss=loss)


class AdaptiveDiceLoss(nn.Module):
    '''
    Integration of DiceLoss on both prob prediction and binary prediction.
    '''

    def __init__(self, eps=1e-6):
        super(AdaptiveDiceLoss, self).__init__()
        from .dice_loss import DiceLoss
        self.main_loss = DiceLoss(eps)
        self.binary_loss = DiceLoss(eps)

    def forward(self, pred, target):
        prob = pred['prob']
        binary = pred['binary']
        gt = target['gt']
        mask = target['mask']
        main_loss = self.main_loss(prob, gt, mask)
        binary_loss = self.binary_loss(binary, gt, mask)
        loss = main_loss + binary_loss
        return loss, dict(main_loss=main_loss, binary_loss=binary_loss)


class AdaptiveInstanceDiceLoss(nn.Module):
    '''
    InstanceDiceLoss on both prob and bianry.
    '''

    def __init__(self, iou_thresh=0.2, thresh=0.3):
        super(AdaptiveInstanceDiceLoss, self).__init__()
        from .dice_loss import InstanceDiceLoss, DiceLoss
        self.main_loss = DiceLoss()
        self.main_instance_loss = InstanceDiceLoss(threshold=thresh, iou_thresh=iou_thresh)
        self.binary_loss = DiceLoss()
        self.binary_instance_loss = InstanceDiceLoss(threshold=thresh, iou_thresh=iou_thresh)
        self.weights = nn.ParameterDict(
            dict(main=nn.Parameter(torch.ones(1)),
                 thresh=nn.Parameter(torch.ones(1)),
                 main_instance=nn.Parameter(torch.ones(1)),
                 binary_instance=nn.Parameter(torch.ones(1))
            )
        )

    def partial_loss(self, weight, loss):
        return loss / weight + 0.5 * torch.log(weight)

    def forward(self, pred, target):
        main_loss = self.main_loss(pred['prob'], target['gt'], target['mask'])
        binary_loss = self.binary_loss(pred['binary'], target['gt'], target['mask'])
        main_instance_loss = self.main_instance_loss(pred['prob'], target['gt'], target['mask'])
        binary_instance_loss = self.binary_instance_loss(pred['binary'], target['gt'], target['mask'])
        loss = self.partial_loss(self.weights['main'], main_loss) \
               + self.partial_loss(self.weights['thresh'], binary_loss) \
               + self.partial_loss(self.weights['main_instance'], main_instance_loss) \
               + self.partial_loss(self.weights['binary_instance'], binary_instance_loss)
        metrics = dict(main_loss=main_loss,
                       binary_loss=binary_loss,
                       main_instance_loss=main_instance_loss,
                       binary_instance_loss=binary_instance_loss)
        
        metrics.update(self.weights)
        return loss, metrics


class L1DiceLoss(nn.Module):
    '''
    L1Loss on thresh, DiceLoss on prob and binary.
    '''

    def __init__(self, eps=1e-6, l1_scale=10):
        super(L1DiceLoss, self).__init__()
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        from .l1_loss import MaskL1Loss
        self.l1_loss = MaskL1Loss()
        self.l1_scale = l1_scale

    def forward(self, pred, target):
        dice_loss, metrics = self.dice_loss(pred, target)
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], target['thresh_map'], target['thresh_mask'])

        loss = dice_loss + self.l1_scale * l1_loss
        metrics.update(**l1_metric)
        return loss, metrics


class FullL1DiceLoss(L1DiceLoss):
    '''
    L1loss on thresh, pixels with topk losses in non-text regions are also counted.
    DiceLoss on binary and prob.
    '''

    def __init__(self, eps=1e-6, l1_scale=10):
        nn.Module.__init__(self)
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        from .l1_loss import BalanceL1Loss
        self.l1_loss = BalanceL1Loss()
        self.l1_scale = l1_scale


class L1BalanceCELoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `prob`,
    MaskL1Loss on `thresh`,
    DiceLoss on `binary`.
    '''
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()
        from .dice_loss import DiceLoss
        from .l1_loss import MaskL1Loss
        from .balance_cross_entropy_loss import BalanceCrossEntropyLossV2
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLossV2()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, target):
        bce_loss = self.bce_loss(pred['prob'], target['gt'], target['mask'])
        metrics = dict(bce_loss=bce_loss)
        if 'thresh' in pred:
            l1_loss, l1_metric = self.l1_loss(pred['thresh'], target['thresh_map'], target['thresh_mask'])
            dice_loss = self.dice_loss(pred['binary'], target['gt'], target['mask'])
            metrics['binary_loss'] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return loss, metrics


class L1BCEMiningLoss(nn.Module):
    '''
    Basicly the same with L1BalanceCELoss, where the bce loss map is used as
        attention weights for DiceLoss
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BCEMiningLoss, self).__init__()
        from .dice_loss import DiceLoss
        from .l1_loss import MaskL1Loss
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, target):
        bce_loss, bce_map = self.bce_loss(pred['prob'], target['gt'], target['mask'], return_origin=True)
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], target['thresh_map'], target['thresh_mask'])
        bce_map = (bce_map - bce_map.min()) / (bce_map.max() - bce_map.min())
        dice_loss = self.dice_loss(pred['binary'], target['gt'], target['mask'], weights=bce_map + 1)
        metrics = dict(bce_loss=bce_loss, binary_loss=dice_loss)
        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics.update(**l1_metric)
        return loss, metrics


class L1LeakyDiceLoss(nn.Module):
    '''
    LeakyDiceLoss on binary, MaskL1Loss on thresh, DiceLoss on binary.
    '''

    def __init__(self, eps=1e-6, coverage_scale=5, l1_scale=10):
        super(L1LeakyDiceLoss, self).__init__()
        from .dice_loss import DiceLoss, LeakyDiceLoss
        from .l1_loss import MaskL1Loss
        self.main_loss = LeakyDiceLoss(coverage_scale=coverage_scale)
        self.l1_loss = MaskL1Loss()
        self.binary_loss = DiceLoss(eps=eps)

        self.l1_scale = l1_scale

    def forward(self, pred, target):
        main_loss, metrics = self.main_loss(pred['prob'], target['gt'], target['mask'])
        binary_loss = self.binary_loss(pred['binary'], target['gt'], target['mask'])
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], target['thresh_map'], target['thresh_mask'])
        metrics.update(**l1_metric, binary_loss=binary_loss)
        loss = main_loss + binary_loss + l1_loss * self.l1_scale
        return loss, metrics
