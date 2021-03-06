import torch
import torch.nn as nn


class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-8):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask, return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred.squeeze(1), gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class BalanceCrossEntropyLossV2(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-8):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask, return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = gt * mask  # float
        negative = (1. - gt) * mask  # float
        positive_count = positive.sum((1, 2))  # (N,)
        negative_count = torch.minimum(negative.sum((1, 2)), positive_count * self.negative_ratio)
        loss = nn.functional.binary_cross_entropy(pred.squeeze(1), gt, reduction='none')
        positive_loss = loss * positive
        negative_loss = loss * negative
        
        balance_loss = 0.
        bsize = positive_count.size(0)
        for i in range(bsize):
            neg_loss, _ = torch.topk(negative_loss[i].view(-1), negative_count[i].int().item())
            balance_loss += (positive_loss[i].sum() + neg_loss.sum()) /\
                (positive_count[i] + negative_count[i] + self.eps)
        balance_loss /= bsize

        if return_origin:
            return balance_loss, loss
        return balance_loss