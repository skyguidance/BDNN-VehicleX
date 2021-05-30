import torch
from torch import nn
from network.backbone import get_backbone
from network.metric_learning import *


class CNN(nn.Module):
    """
    CNN Feature Extractor
    """

    def __init__(self, config, loss_module="softmax"):
        super(CNN, self).__init__()
        self.backbone, self.backbone_output_dim = get_backbone(config)
        self.loss_module = loss_module
        if self.loss_module == "softmax":
            self.fc = nn.Linear(self.backbone_output_dim, config["model"]["output_dims"])
        elif self.loss_module == "arcface":
            self.fc = ArcMarginProduct(self.backbone_output_dim, config["model"]["output_dims"], s=30, m=0.5,
                                       easy_margin=False, ls_eps=0.0)
        elif self.loss_module == "cosface":
            self.fc = ArcMarginProduct(self.backbone_output_dim, config["model"]["output_dims"], s=30, m=0.5)
        else:
            raise NotImplementedError
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, label=None, device="cpu"):
        x = self.forward_backbone(x)
        if self.loss_module == "softmax":
            x = self.fc(x)
        elif self.loss_module in ["arcface", "cosface"]:
            x = self.fc(x, label)
        else:
            raise NotImplementedError
        return x

    def forward_backbone(self, x):
        x = self.backbone(x)
        # Needs GAP.
        if len(x.shape) == 4 and x.shape[3] != 1:
            x = self.gap(x)
        x = x.flatten(1)
        return x
