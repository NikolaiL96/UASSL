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
    def __init__(self, rep_dim=2048, use_bias=True):
        super().__init__()

        self.kappa_net = nn.Sequential(
            nn.Linear(rep_dim, rep_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(rep_dim, 1, bias=use_bias),
        )

    def forward(self, x):
        kappa = self.kappa_net(x)
        return kappa


class ProbabilisticLayer(nn.Module):
    def __init__(self, distribution,
                 in_features,
                 use_bias: bool = True,
                 eps: float = 1e-4):
        super().__init__()

        self.eps = eps
        self.in_features = in_features
        self.distribution = distribution

        if distribution in ["powerspherical", "normal", "sphere"]:
            self.layer = nn.Linear(in_features, in_features, bias=use_bias)
        elif distribution in ["sphereNoFC"]:
            self.layer = nn.Identity()

    def forward(self, x, unc=None):
        mean = self.layer(x)

        #const = torch.pow(torch.tensor(self.in_features).to(mean.device), 1 / 2.)
        const = torch.linalg.norm(mean, dim=1)
        kappa = nn.functional.softplus(unc.squeeze())
        kappa = const * kappa + self.eps

        if self.distribution == "powerspherical":
            mean = nn.functional.normalize(mean, dim=1)
            return powerspherical.PowerSpherical(mean, kappa)

        if self.distribution == "vonMisesFisher":
            mean = nn.functional.normalize(mean, dim=1)
            return vonmisesfisher.VonMisesFisher(mean, kappa)

        if self.distribution == "normal":
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(1).repeat(1, self.in_features)
            return normaldistribution.Normal(mean, kappa)

        if self.distribution == "normalSingleScale":
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(1).repeat(1, self.in_features)
            return normaldistribution.Normal(mean, kappa)

        if self.distribution == "sphere" or self.distribution == "sphereNoFC":
            norm_mean = torch.linalg.norm(mean, dim=1, keepdim=True)
            return pointdistribution.PointDistribution(loc=mean / norm_mean, scale=norm_mean)

        else:
            raise Exception("Distribution not implemented")


class UncertaintyNet(nn.Module):
    def __init__(self, in_channels, distribution_type, network: str = "resnet50", checkpoint_path=None,
                 eps: float = 1e-4):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_path = checkpoint_path
        self.eps = eps
        self.in_channels = in_channels
        self.network = network
        self.name = "UncertaintyNet"

        # Construct mean_net and kappa_net models
        self.model = self._build_model()
        rep_dim = self.model.fc.in_features

        self.kappa_model = KappaNet(rep_dim=rep_dim)
        self.fc = ProbabilisticLayer(distribution_type, rep_dim)
        self.model.fc = nn.Identity()

        # Load and initialize pre-trained model parameters
        if checkpoint_path is not None:
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

        dist = self.fc(feats, unc)
        return dist
