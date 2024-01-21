import torch.nn as nn
from torchvision.models import resnet18, resnet50


def cifar_resnet18(in_channels: int = 3):
    resnet = resnet18(zero_init_residual=True)

    # Cifar specifics
    resnet.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
    resnet.maxpool = nn.Identity()
    resnet.name = "ResNet18"
    return resnet

def cifar_resnet50(in_channels: int = 3):
    resnet = resnet50(zero_init_residual=True)

    # Cifar specifics
    resnet.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
    resnet.maxpool = nn.Identity()
    resnet.name = "ResNet50"
    return resnet