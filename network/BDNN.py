import torch
from torch import nn


class BiDirectionalNN(nn.Module):
    """
    BDNN
    """

    def __init__(self):
        super(BiDirectionalNN, self).__init__()
        self.net = nn.Sequential()

    def forward(self, x):
        x = self.net(x)
        return x
