# We implement a regularizer that works with all implemented distributions:
import torch
import torch.nn as nn
import math
from .powerspherical import PowerSpherical, HypersphericalUniform, _kl_powerspherical_uniform
from .vonmisesfisher import HypersphericalUniform, VonMisesFisher, _kl_vmf_uniform


class Probabilistic_Regularizer(nn.Module):
    def __init__(self, distribution, lambda_reg, eps: float = 1e-6):
        super(Probabilistic_Regularizer, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.distribution = distribution
        self.lambda_reg = lambda_reg.to(self.device)

        print(
            f"Init Probabilistic Reguarlizer with `{self.distribution}` and lambda_reg={self.lambda_reg}"
        )
        self.eps = eps


    def forward(self, x):
        dist1, dist2 = x

        if self.distribution == "normal" or self.distribution == "normalSingleScale":
            return self.two_normal_reg(dist1, dist2)

        elif self.distribution in ["powerspherical"]:
            return self.power_spherical_reg(dist1) + (self.power_spherical_reg(dist2))

        elif self.distribution == "vonMisesFisherNorm" or self.distribution == "vonMisesFisherNode":
            return self.vmf_reg(dist1) + (self.vmf_reg(dist2))

        elif self.distribution == "sphere":
            return torch.zeros(1, device=self.device)

        else:
            return torch.zeros(1, device=self.device)


    def KL_two_normal(self, x1, x2, reduction='mean'):
        if reduction == 'none':
            mean1, mean2 = x1.mean.unsqueeze(1), x2.mean.unsqueeze(0)
            logvar1, logvar2 = x1.logvar.unsqueeze(1), x2.logvar.unsqueeze(0)
        else:
            mean1, mean2 = x1.mean, x2.mean
            logvar1, logvar2 = x1.logvar, x2.logvar

        # KL Divergenz of two normal distributions with diagonal variance
        KLD = 0.5 * torch.sum(logvar2 - logvar1 - 1 +
                              ((mean1 - mean2).pow(2) + logvar1.exp()).div(logvar2.exp() + self.eps),
                              dim=-1)

        return KLD.mean()

    def two_normal_reg(self, x1, x2, reduction='mean'):
        KLD1 = self.KL_two_normal(x1, x2, reduction)
        KLD2 = self.KL_two_normal(x2, x1, reduction)

        return self.lambda_reg * 0.5 * (KLD1 + KLD2)

    def power_spherical_reg(self, x):
        dim = x.loc.shape[-1]

        prior = HypersphericalUniform(dim, device=self.device)
        objective = self.lambda_reg * _kl_powerspherical_uniform(x, prior)

        return objective.mean()

    def vmf_reg(self, x):
        dim = x.mean.shape[-1]
        prior = HypersphericalUniform(dim, device=self.device)
        # TODO Why do we not just regularize the entropy when the kl is computed with a simple addition of entropy to the constant entropy of the prio?
        return self.lambda_reg * _kl_vmf_uniform(x, prior).mean()