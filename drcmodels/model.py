# -*- coding: utf-8 -*-

from addict import Dict
from torch import nn
import torch.nn.functional as F

from drcmodels.backbone import build_backbone
from drcmodels.neck import build_neck
from drcmodels.head import build_head


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        :param model_config: 模型配置
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size() # _, _, H, W = batch, channel=3, height, width
        backbone_out = self.backbone(x)
        # print(backbone_out)
        neck_out = self.neck(backbone_out)
        # print(neck_out.shape)
        y = self.head(neck_out)
        # print(y.shape)
        # y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    import torch


