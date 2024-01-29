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
    def __init__(self, rep_dim=2048, use_bias=True, hidden_dim=2048):
        super().__init__()

        # self.kappa_net = nn.Sequential(
        #     nn.Linear(rep_dim, rep_dim, bias=use_bias),
        #     nn.BatchNorm1d(num_features=rep_dim),
        #     nn.ReLU(),
        #     nn.Linear(rep_dim, 1, bias=use_bias),
        # )

        self.kappa_net = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim, bias=use_bias),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=use_bias),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=use_bias),
        )

    def forward(self, x):
        return self.kappa_net(x)



class ProbabilisticLayer(nn.Module):
    def __init__(self, distribution,
                 in_features,
                 use_bias: bool = True,
                 eps: float = 1e-4):
        super().__init__()

        self.eps = eps
        self.in_features = in_features
        self.distribution = distribution

        if distribution in ["powerspherical", "normal", "normalSingleScale", "sphere", "vonMisesFisher"]:
            self.layer = nn.Linear(in_features, in_features, bias=use_bias)
        elif distribution in ["sphereNoFC", "powersphericalNoFC", "normalSingleScaleNoFC", "normalNoFC",
                              "vonMisesFisherNoFC"]:
            self.layer = nn.Identity()

    def forward(self, x, unc):
        mean = self.layer(x)

        if self.distribution in ["sphere", "sphereNoFC"]:
            norm_mean = torch.linalg.norm(mean, dim=1, keepdim=True)
            return pointdistribution.PointDistribution(loc=mean / norm_mean, scale=norm_mean)

        const = torch.pow(torch.tensor(self.in_features).to(mean.device), 1 / 2.)
        kappa = nn.functional.softplus(unc.squeeze())
        kappa = const * kappa + self.eps

        if "powerspherical" in self.distribution:
            mean = nn.functional.normalize(mean, dim=1)
            return powerspherical.PowerSpherical(mean, kappa)

        if "vonMisesFisher" in self.distribution:
            mean = nn.functional.normalize(mean, dim=1)
            return vonmisesfisher.VonMisesFisher(mean, kappa)

        if "normal" in self.distribution:
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(1).expand(-1, self.in_features)
            return normaldistribution.Normal(mean, kappa)

        if self.distribution == "normalSingleScale":
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(1).expand(-1, self.in_features)
            return normaldistribution.Normal(mean, kappa)

        else:
            raise Exception("Distribution not implemented")


class UncertaintyNet(nn.Module):
    def __init__(self, in_channels, distribution_type, checkpoint_path=None, eps: float = 1e-4):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_path = checkpoint_path
        self.eps = eps
        self.in_channels = in_channels
        self.name = "UncertaintyNet"

        # Construct mean_net and kappa_net models
        self.mean_model = self._build_model()
        rep_dim = self.mean_model.fc.in_features

        # No need to train a kappa model for non-probabilistic baseline.
        if distribution_type not in ["sphere", "sphereNoFC"]:
            self.kappa_model = KappaNet(rep_dim=rep_dim)
        else:
            self.kappa_model = nn.Identity()

        self.fc = ProbabilisticLayer(distribution_type, rep_dim)
        self.mean_model.fc = nn.Identity()

        # Load and initialize pre-trained model parameters
        if checkpoint_path is not None:
            self._load_params()

    def _load_params(self):
        # TODO write checkpoint function that automatically loads pretrained baseline if fine tune flag is set.
        # Load pre-trained model for the mean_ and kappa_net
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        params, self.projector_params = _get_state_dict(checkpoint["model"])
        msg = self.mean_model.load_state_dict(params, strict=False)
        # msg = self.kappa_model.load_state_dict(params, strict=False)

    def _build_model(self):
        resnet = resnet18(zero_init_residual=True)

        resnet.conv1 = nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False)
        resnet.maxpool = nn.Identity()
        return resnet

    def forward(self, x):
        feats = self.mean_model(x)
        unc = self.kappa_model(feats.detach())

        return self.fc(feats, unc)

