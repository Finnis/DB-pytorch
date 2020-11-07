from .seg_detector import SegDetector
from .resnet import resnet18, resnet34, resnet50, resnet101, deformable_resnet50, deformable_resnet18

import torch.nn as nn


class DBModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = eval(cfg['arch']['backbone']['type'])(**cfg['arch']['backbone']['args'])
        self.decoder = eval(cfg['arch']['head']['type'])(**cfg['arch']['head']['args'])
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)

        return x