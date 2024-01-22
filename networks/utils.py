import torch
import torch.nn as nn

# Code from facebook ConvNext implementation:
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L119
class LayerNorm2D(nn.Module):
    def __init__(self, num_features: int, affine: bool=True, eps: float=1e-6):
        '''
        input shape: (N,C,H,W) or (C,H,W)
        '''
        super().__init__()
        self.weight = torch.ones(num_features)
        self.bias = torch.zeros(num_features)
        if affine: # Turn into trainable parameters
            self.weight = nn.Parameter(self.weight)
            self.bias = nn.Parameter(self.bias)
        self.eps = eps
    
    def forward(self, x):
        if x.dim()==4:
            dim = 1
        elif x.dim()==3:
            dim = 0
        else:
            raise ValueError('Input format must be either (N,C,H,W) or (C,H,W).')
            
        u = x.mean(dim, keepdim=True)
        s = (x - u).pow(2).mean(dim, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# Define callable norm layer
norm_layers = {'batch': nn.BatchNorm2d,
               'instance': nn.InstanceNorm2d,
               'layer': LayerNorm2D,
               'none': None}

# Define callable activations
activations = {'relu': nn.ReLU(inplace=True),
               'gelu': nn.GELU(),
               'selu': nn.SELU(inplace=True),
               'elu': nn.ELU(inplace=True),
               'none': None}

# Define callable poolings
poolings2D = {'max': nn.MaxPool2d,
              'average': nn.AvgPool2d,
              'ada_max': nn.AdaptiveMaxPool2d,
              'ada_average': nn.AdaptiveAvgPool2d,
              'none': None}