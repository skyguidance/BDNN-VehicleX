from torch import nn


class SimpleNN(nn.Module):
    """
    A simple NN used as baseline.
    """

    def __init__(self, mlp_layers, input_dims, output_dims, dropout_rate, do_BN=False):
        super(SimpleNN, self).__init__()
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
        return x
