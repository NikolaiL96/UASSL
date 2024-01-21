# We implement a regularizer that works with all implemented distributions:
import torch
import torch.nn as nn
import math
from .powerspherical import PowerSpherical, HypersphericalUniform, _kl_powerspherical_uniform
from .vonmisesfisher import HypersphericalUniform, VonMisesFisher, _kl_vmf_uniform


class Probabilistic_Regularizer(nn.Module):
    def __init__(self, distribution, lambda_reg, eps: float = 1e-6, repre_dim=512):
        super(Probabilistic_Regularizer, self).__init__()
        self.distribution = distribution
        self.lambda_reg = lambda_reg
        self.repre_dim = repre_dim
        print(
            f"Init Probabilistic Reguarlizer with `{self.distribution}` and lambda_reg={self.lambda_reg}"
        )
        self.eps = eps
        if "vade" in distribution:
            try:
                n_classes = int("".join(filter(str.isdigit, distribution)))
            except:
                n_classes = 10
            self.init_vade_regularization(n_classes=n_classes)

    def forward(self, x, reduction='mean', mask=None):
        if isinstance(x, tuple):
            dist1, dist2 = x
        else:
            dist1 = x
            dist2 = None
        if self.distribution == "normal" or self.distribution == "normalSingleScale":
            if isinstance(x, tuple):
                return self.two_normal_reg(*x, reduction)
            else:
                return self.normal_reg(dist1) + (self.normal_reg(dist2) if dist2 is not None else 0)
        if self.distribution == "entropyregnormal":
            return -1 * torch.mean(dist1.entropy() + dist2.entropy()) * self.lambda_reg

        if self.distribution in ["powerspherical", "powerspherical_wo_fc_lin","powerspherical_wo_fc_nonlin", "powerspherical_wo_fc_reuse"]:
            return self.power_spherical_reg(dist1) + (self.power_spherical_reg(dist2))

        if self.distribution == "vonMisesFisherNorm" or self.distribution == "vonMisesFisherNode":
            # Either parametrized by norm or output-node
            return self.vmf_reg(dist1) + (self.vmf_reg(dist2) if dist2 is not None else 0)
        if "MixtureOfGaussians" in self.distribution:
            return self.mixture_experts_reg(dist1) + (self.mixture_experts_reg(dist2) if dist2 is not None else 0)
        if self.distribution == "" or self.distribution == "sphere":
            return torch.zeros((1), device=dist1.mean.device)
        if "vade" in self.distribution:
            if "DC" in self.distribution:
                if dist2 is None:
                    raise ValueError("We assume to have to Distributions for the Constrained clustering")
                return self.vade_regularization(dist1, dist2)
            return self.vade_regularization(dist1) + (self.vade_regularization(dist2) if dist2 is not None else 0)

        return torch.zeros((1), device=dist1.mean.device)

    def normal_reg(self, x):

        standard_normal = torch.distributions.Normal(torch.zeros_like(x.mean), torch.ones_like(x.mean))

        return self.lambda_reg * torch.distributions.kl_divergence(x, standard_normal).mean()

    # ToDo Think if can be deleted
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

        if reduction == 'mean':
            return KLD.mean()
        else:
            return KLD

    def two_normal_reg(self, x1, x2, reduction='mean'):
        # symmetric KL Diverence of two normal distributions
        KLD1 = self.KL_two_normal(x1, x2, reduction)
        KLD2 = self.KL_two_normal(x2, x1, reduction)

        return self.lambda_reg * 0.5 * (KLD1 + KLD2)

    def power_spherical_reg(self, x):
        dim = x.loc.shape[-1]
        device = x.loc.device
        prior = HypersphericalUniform(dim, device)
        objective = self.lambda_reg * _kl_powerspherical_uniform(x, prior)

        return objective.mean()

    def vmf_reg(self, x):
        dim = x.mean.shape[-1]
        device = x.mean.device
        prior = HypersphericalUniform(dim, device)
        # TODO Why do we not just regularize the entropy when the kl is computed with a simple addition of entropy to the constant entropy of the prio?
        return self.lambda_reg * _kl_vmf_uniform(x, prior).mean()

    def mixture_experts_reg(self, x):
        # Regularize each component to unit normal?
        # we assume a gaussian component and regularize the choosen component by mean?
        return (
                self.lambda_reg
                * -0.5
                * torch.mean(
            x.mixture_distribution.probs
            * (
                    1
                    + torch.log(x.component_distribution.scale ** 2 + self.eps)
                    - x.component_distribution.mean.pow(2)
                    - x.component_distribution.scale ** 2
            )
        )
        )

    def init_vade_regularization(self, n_classes=10):
        """
        Initialize a Mixture of Gaussian Prior for the Latent Distribution
        Inspired by Variational Deep Embedding (https://arxiv.org/pdf/1611.05148.pdf)
        """

        self.means = nn.Parameter(torch.rand((n_classes, 1, self.repre_dim)) * 2 - 1, requires_grad=True)
        self.variances_raw = nn.Parameter(torch.zeros(n_classes, 1, self.repre_dim), requires_grad=True)
        self.pi = nn.Parameter(torch.ones((n_classes, 1, 1)) / n_classes, requires_grad=False)

    def vade_regularization(self, x, y=None):
        """
        Regularize the latent distribution with a Mixture of Gaussian Prior
        """
        # We regularize to minimize the KL to the Mixture of Gaussian according to a Generative Model
        # variances = torch.exp(self.variances_logits)
        variances = torch.nn.functional.softplus(self.variances_raw) + self.eps
        variances_logits = torch.log(variances)
        pi = torch.nn.functional.softmax(self.pi, dim=0)
        prior_dist = torch.distributions.Normal(self.means, variances ** 0.5)

        Z = x.rsample()
        x_scale = x.scale
        x_mean = x.mean
        x_variance = x.variance
        if y is not None:
            Z = torch.cat([Z, y.rsample()], dim=0)
            x_mean = torch.cat([x_mean, y.mean], dim=0)
            x_scale = torch.cat([x_scale, y.scale], dim=0)
            x_variance = torch.cat([x_variance, y.variance], dim=0)

        # Z = x.mean

        z_dim = x.mean.shape[-1]
        p_c_z = torch.exp(
            torch.mean(torch.log(pi) + torch.clamp(prior_dist.log_prob(Z.unsqueeze(0)), -1e5), dim=-1)) + self.eps
        gamma = p_c_z / torch.sum(p_c_z, dim=0, keepdim=True)

        # E q(z,c|x)  log(p(z|c))
        h = (math.log(math.sqrt(2 * math.pi)) + \
             0.5 * torch.mean(variances_logits, dim=-1) + \
             0.5 * torch.mean(x_variance.unsqueeze(0) / variances, dim=-1) + \
             0.5 * torch.mean((x_mean.unsqueeze(0) - self.means) ** 2 / variances, dim=-1))

        l1 = - torch.mean(gamma * h)

        # E q(z,c|x) log(p(c))
        l2 = torch.mean(gamma * torch.log(pi))

        # E q(z,c|x) log(q(z|x))
        l3 = torch.mean((-1 * z_dim * math.log(math.sqrt(2 * math.pi)) - \
                         0.5 * torch.mean(1 + 2 * torch.log(x_scale + self.eps), dim=-1)))

        # E q(z,c|x) log(q(c|x))
        l4 = torch.mean(torch.xlogy(gamma, gamma + 1e-5))

        # Originally, q(c|x) is computed by E q(z|x)  p(c|z)
        # = p(c)*p(z|c) / (sum(k=1-K) p(c_k) p(z|c))

        # We also use the pairwise constraints to force two augmentations belonging to the same cluster
        if y is not None:
            gamma1, gamma2 = torch.chunk(gamma, 2, dim=1)
            l5 = torch.mean(gamma1 * gamma2)
        else:
            l5 = 0
        # print(f"L1 {l1}, L2 {l2}, L3 {l3}, L4 {l4}, Gamma {gamma[:,0]}")
        return -1 * self.lambda_reg * (l1 + l2 - l3 - l4 + l5)  # + l3+ l4 )