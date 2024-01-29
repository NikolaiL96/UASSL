import torch
import torch.nn as nn

from distributions import (
    pointdistribution,
    powerspherical,
    vonmisesfisher,
    normaldistribution,
)


class Probabilistic_Layer(nn.Module):
    def __init__(self, distribution: str = "none",
                 in_features: int = 1000,
                 use_bias: bool = True,
                 eps: float = 1e-4):
        super().__init__()

        self.dim = in_features
        self.distribution = distribution

        if 'powerspherical' in distribution:
            out_features = in_features + 1
        elif distribution == "normal":
            out_features = in_features * 2
        elif distribution in ["vonMisesFisher", "normalSingleScale"]:
            out_features = in_features + 1
        elif distribution in ["sphere", "sphereNoFC"]:
            out_features = in_features
        else:
            out_features = in_features

        if distribution != "sphereNoFC":
            self.layer = nn.Linear(in_features, out_features, bias=use_bias)
        else:
            self.layer = nn.Identity()

        self.eps = eps

    def forward(self, x):

        if self.distribution == "normal":
            mean, logvar = torch.chunk(self.layer(x), 2, dim=1)
            std = nn.functional.softplus(logvar) + self.eps

            out_dist = normaldistribution.Normal(mean, std)
            return out_dist

        if self.distribution == "normalSingleScale":
            feats = self.layer(x)
            mean = nn.functional.normalize(feats[:, :self.dim], dim=1)
            std = nn.functional.softplus(feats[:, -1]) + self.eps

            out_dist = normaldistribution.Normal(mean, std.unsqueeze(1).expand(-1, mean.shape[1]))
            return out_dist

        if self.distribution in ["sphere", "sphereNoFC"]:
            mu = self.layer(x)
            norm_mu = torch.linalg.norm(mu, dim=1, keepdim=True)
            return pointdistribution.PointDistribution(mu / norm_mu, norm_mu.squeeze())

        if self.distribution == "powerspherical":
            feats = self.layer(x)
            mu = nn.functional.normalize(feats[:, :self.dim], dim=1)

            const = torch.pow(torch.tensor(self.dim).to(mu.device), 1 / 2.)
            kappa = const * nn.functional.softplus(feats[:, -1]) + self.eps
            return powerspherical.PowerSpherical(mu, kappa)

        if self.distribution == "vonMisesFisher":
            feats = self.layer(x)
            mu = nn.functional.normalize(feats[:, :self.dim], dim=1)

            const = torch.pow(torch.tensor(self.dim).to(mu.device), 1 / 2.)
            kappa = const * nn.functional.softplus(feats[:, -1]) + self.eps
            return vonmisesfisher.VonMisesFisher(mu, kappa)

        raise ValueError('Forward pass with unknown distribution.')
