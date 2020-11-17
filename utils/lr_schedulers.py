"""Popular Learning Rate Schedulers"""
import torch
import numpy as np

__all__ = ['WarmupPolyLR', 'WarmupCosineAnnealingLR']


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):
        assert warmup_method in ("constant", "linear")
        
        self.last_lr = last_lr
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.Tmax = max_iters - warmup_iters

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        Tcur = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.last_lr + (base_lr - self.last_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - Tcur / self.Tmax, self.power)
        return [self.last_lr + (base_lr - self.last_lr) * factor for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_lr=0, Tmax=0, warmup_factor=1.0 / 3,
                 warmup_iters=0, warmup_method='linear', last_epoch=-1):
        assert warmup_method in ("constant", "linear")
        
        self.last_lr = last_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.Tmax = Tmax

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        Tcur = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.last_lr + (base_lr - self.last_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = 1 + np.cos((Tcur % self.Tmax) / self.Tmax * np.pi)
        return [self.last_lr + 0.5 * (base_lr - self.last_lr) * factor for base_lr in self.base_lrs]


if __name__ == '__main__':
    import torch
    from torchvision.models import resnet18

    max_iter = 12
    model = resnet18()
    op = torch.optim.SGD(model.parameters(), 1.)
        
    sc = WarmupPolyLR(op, max_iters=max_iter, power=0.9,
                      warmup_iters=3, warmup_method='linear',
                      last_epoch=-1, last_lr=0.5)
    print(sc.last_epoch, sc.get_lr()[0])
    input()
    lr = []
    for i in range(max_iter):
        sc.step()
        print(i, sc.last_epoch, sc.get_lr()[0])
        input()
        lr.append(sc.get_lr()[0])
    
    # from matplotlib import pyplot as plt
    # plt.plot(list(range(max_iter)), lr)
    # plt.show()
