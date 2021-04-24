import torch
from torch import nn


class BiDirectionalNN(nn.Module):
    """
    BDNN
    """

    def __init__(self, mlp_layers, input_dims, output_dims, dropout_rate, do_BN=False):
        super(BiDirectionalNN, self).__init__()
        mlp = nn.Sequential()
        # Standard MLP with ReLU, Dropout and BN as optional.
        for (i, out_dims) in enumerate(mlp_layers):
            mlp.add_module(f"layer{i}", nn.Linear(input_dims, out_dims))
            if do_BN:
                mlp.add_module(f"bn{i}", nn.BatchNorm1d(out_dims))
            mlp.add_module(f"relu{i}", nn.ReLU())
            mlp.add_module(f"dropout{i}", nn.Dropout(dropout_rate))
            input_dims = out_dims
        # Added final Classification Layer.
        mlp.add_module(f"final", nn.Linear(input_dims, output_dims))
        self.net = mlp

    def forward(self, x):
        x = self.net(x)
        return x[:, :-1], x[:, -1]


class BiDirectionalNN_R(nn.Module):
    """
    Reversed BDNN
    """

    def __init__(self, mlp_layers, input_dims, output_dims, dropout_rate, do_BN=False):
        super(BiDirectionalNN_R, self).__init__()
        mlp = nn.Sequential()
        # Reversed Model of original BiDirectionalNN.
        mlp.add_module(f"final", nn.Linear(output_dims, mlp_layers[-1]))
        for (i, out_dims) in enumerate(mlp_layers[::-1]):
            idx = len(mlp_layers) - 1 - i
            mlp.add_module(f"dropout{idx}", nn.Dropout(dropout_rate))
            mlp.add_module(f"relu{idx}", nn.ReLU())
            if do_BN:
                mlp.add_module(f"bn{idx}", nn.BatchNorm1d(out_dims))
            if i < len(mlp_layers) - 1:
                mlp.add_module(f"layer{idx}", nn.Linear(out_dims, mlp_layers[::-1][i + 1]))
            else:
                mlp.add_module(f"layer{idx}", nn.Linear(out_dims, input_dims))
        self.net = mlp

    def forward(self, x):
        x = self.net(x)
        return x
