"""
The purpose of this script is to trial-run evaluation.covspectral.
It can be served as a prototype of how use the evaluator in the training script.
"""

import numpy as np
from matplotlib import pyplot as plt

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

import models

import eval

##### the code below is just for preparing the model
data_root = "./data"

# Cifar10 Mean and Std
CIFAR10_NORM = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

dl_kwargs = {"batch_size": 512, "shuffle": True, "num_workers": 2}

# this is important
distribution_type = ""

# these two are only important for training
beta = 0.1
loss_type = "barlowtwins"

data_dir = "./VariationSSL15-12-2021-13-04-16/variational_barlow_twins-squared-norm-0.1"
checkpoint = torch.load(f"{data_dir}/init_model.tar")

resnet = resnet18(zero_init_residual=True)
repre_dim = resnet.fc.in_features

test_transf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(*CIFAR10_NORM)]
)
test_ds = CIFAR10(root=data_root, train=False, transform=test_transf, download=True)
test_dl = DataLoader(test_ds, drop_last=False, **dl_kwargs)

barlow_twins = models.VariationalBarlowTwins(
    resnet,
    beta=beta,
    projector_hidden=(2048, 2048),
    distribution_type=distribution_type,
    loss_type=loss_type,
)

barlow_twins.load_state_dict(checkpoint["model"])

##### actual code for evaluation here

covspectral = eval.CovarianceSpectral()

eigvals = covspectral.eval(test_dl, barlow_twins)

plt.title("Eigenvalues")

plt.plot(eigvals)
plt.ylabel("eigenvalue (log scale)")
plt.yscale("log")
plt.xlabel("Rank")
plt.savefig(f"{data_dir}/eigenspectrum.png")

print("Evaluation Done!!!")
