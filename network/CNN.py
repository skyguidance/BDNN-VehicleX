import torch
from torch import nn
from network.backbone import get_backbone


class CNN(nn.Module):
    """
    CNN Feature Extractor
    """

    def __init__(self, config):
        super(CNN, self).__init__()
        self.backbone, self.backbone_output_dim = get_backbone(config)
        self.fc = nn.Linear(self.backbone_output_dim, config["model"]["output_dims"])

    def forward(self, x):
        x = self.forward_backbone(x)
        x = self.fc(x)
        return x

    def forward_backbone(self, x):
        x = self.backbone(x)
        return x
