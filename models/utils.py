from typing import Union

import torch.nn as nn
#import torchadf.nn as adf
from torchadf.nn import Sequential, Linear, ReLU


class MLP(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_dims: Union[int, tuple],
                 bias: bool = True,
                 use_batchnorm=True,
                 batchnorm_last: bool = False):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        mlp = [nn.Linear(in_dim, hidden_dims[0], bias=bias)]
        for i in range(len(hidden_dims) - 1):
            if use_batchnorm:
                mlp.append(nn.BatchNorm1d(hidden_dims[i]))
            mlp.extend([nn.ReLU(inplace=True),
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias)])
        if batchnorm_last:
            mlp.append(nn.BatchNorm1d(hidden_dims[-1]))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)

class MLP_variational(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_dims: Union[int, tuple],
                 bias: bool = True,):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        mlp = [Linear(in_dim, hidden_dims[0], bias=bias, mode="diag")]
        for i in range(len(hidden_dims) - 1):
            mlp.extend([ReLU(mode="diag"),
                        Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias, mode="diag")])

        self.mlp = Sequential(*mlp)

    def forward(self, mean, var):
        return self.mlp(mean, var)
