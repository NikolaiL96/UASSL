import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet50

from utils.utils import _get_state_dict
from distributions import (
    pointdistribution,
    powerspherical,
    vonmisesfisher,
    normaldistribution,
)


class KappaNet(nn.Module):
    def __init__(self, in_channels, rep_dim=512, use_bias=True):
        super().__init__()

        self.kappa_net = nn.Sequential(
            nn.Linear(in_channels, rep_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(rep_dim, 1, bias=use_bias),
        )

    def forward(self, x):
        x = self.kappa_net(x)
        return x


class ProbabilisticLayer(nn.Module):
    def __init__(self, distribution,
                 rep_dim,
                 use_bias: bool = True,
                 eps: float = 1e-4):
        super().__init__()

        self.eps = eps
        self.rep_dim = rep_dim
        self.distribution = distribution

        if distribution in ["powerspherical", "normal", "sphere", "powerspherical_2048"]:
            self.layer = nn.Linear(rep_dim, rep_dim, bias=use_bias)
        elif "powersphericaln" in distribution:
            self.div = int("".join(filter(str.isdigit, distribution)))
            self.layer = nn.Linear(rep_dim, rep_dim, bias=use_bias)
        elif distribution in ["sphereNoFC"]:
            self.layer = nn.Identity()

    def forward(self, x, unc=None):
        mean = self.layer(x)

        if unc is not None:
            kappa = nn.functional.softplus(unc.squeeze())
            const = torch.pow(torch.tensor(self.rep_dim).to(kappa.device), 1 / 2.)
        else:
            raise Exception("Please provide an uncertainty estimate.")

        kappa = const * kappa + self.eps

        if self.distribution == "powerspherical":
            mean = nn.functional.normalize(mean, dim=1)
            return powerspherical.PowerSpherical(mean, kappa)

        if self.distribution == "normal":
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(1).repeat(1, self.rep_dim)
            return normaldistribution.Normal(mean, kappa)

        if self.distribution == "normalSingleScale":
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(1).repeat(1, self.rep_dim)
            return normaldistribution.Normal(mean, kappa)

        if self.distribution == "sphere" or self.distribution == "sphereNoFC":
            norm_mean = torch.linalg.norm(mean, dim=1, keepdim=True)
            return pointdistribution.PointDistribution(loc=mean / norm_mean, scale=kappa)

        else:
            raise Exception("Distribution not implemented")


class UncertaintyNet(nn.Module):
    def __init__(self, in_channels, distribution_type, checkpoint_path=None, eps: float = 1e-4,
                 pretrained: bool = False, network: str = "resnet50"):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels

        self.checkpoint_path = checkpoint_path
        self.eps = eps
        self.pretrained = pretrained
        self.network = network
        self.name = "UncertaintyNet"

        # Construct mean_net and kappa_net models
        self.model = self._build_model()
        self.kappa_model = KappaNet(in_channels=self.in_channels, rep_dim=self.model.fc.in_features())
        self.probabilistic_layer = ProbabilisticLayer(distribution_type, self.model.fc.in_features())
        self.model.fc = nn.Identity()

        # Load and initialize pre-trained model parameters
        if self.pretrained and self.network == "resnet18":
            self._load_params()

    def _load_params(self):
        # Load pre-trained model for the mean_ and kappa_net
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        params, self.projector_params = _get_state_dict(checkpoint["model"])
        msg = self.model.load_state_dict(params, strict=False)
        # msg = self.kappa_model.load_state_dict(params, strict=False)

    def _build_model(self):
        if self.network == "resnet18":
            resnet = resnet18(zero_init_residual=True)
        elif self.network == "resnet50":
            resnet = resnet50(zero_init_residual=True)


        resnet.conv1 = nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False)
        resnet.maxpool = nn.Identity()

        return resnet

    def forward(self, x):
        feats = self.model(x)
        unc = self.kappa_model(feats.detach())

        dist = self.probabilistic_layer(feats, unc)
        return dist
