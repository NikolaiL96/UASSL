from typing import Union
import warnings

import torch
import torch.nn as nn
from distributions import Probabilistic_Layer, Probabilistic_Regularizer, Normal
from .utils import MLP, MLP_variational
from losses import MCNTXent, NTXent, KL_Loss, UncertaintyLoss


class SimCLR(nn.Module):
    # @ex.capture
    def __init__(
            self,
            backbone_net: nn.Module,
            projector_hidden: Union[int, tuple] = (2048, 2048, 256),
            loss: str = "NT-Xent",
            temperature: float = 0.05,
            distribution_type: str = None,
            lambda_reg: float = 0.001,
            lambda_unc: float = 0.,
            n_mc: int = 16,
            loc_warmup: int = 0,
    ):
        super().__init__()

        self.loss_fn = None
        self.projector = None
        self.backbone_net = backbone_net
        self.rep_dim = self.backbone_net.fc.in_features

        if backbone_net.name != "UncertaintyNet":
            backbone_net.fc = nn.Identity()
            self.backbone_net.fc = Probabilistic_Layer(distribution_type, in_features=self.rep_dim)
        self.projector_hidden = projector_hidden

        self.loss = loss
        self.temperature = temperature
        self.lambda_unc = lambda_unc
        self.loc_warmup = loc_warmup
        self.n_mc = n_mc
        self.distribution_type = distribution_type

        self.initialize_projector(projector_hidden)
        self.initialize_loss(loss, temperature, n_mc)
        # Regularizer for the generated distribution
        self.regularizer = Probabilistic_Regularizer(distribution_type, lambda_reg)

        if self.lambda_unc != 0.:
            self.uncertainty_loss = UncertaintyLoss(lambda_unc)
            print("We use the Uncertainty Loss")

        # Verbose
        print(f"We initialize SimCLR with {self.rep_dim} dimensions and a {distribution_type} distribution.")


    def initialize_projector(self, projector_hidden):
        if projector_hidden:
            if (self.loss == "KL_Loss") and (self.distribution_type == "normal"):
                self.projector = MLP_variational(self.rep_dim, projector_hidden, bias=False)
                print(f"The variational projector has {projector_hidden} hidden units")
            elif (self.loss == "KL_Loss") and (self.distribution_type != "normal"):
                self.projector = nn.Identity()
                warnings.warn("Projector hidden with KL Loss and no Normal distribution specified. No projection "
                              "head will be used!")
            else:
                self.projector = MLP(self.rep_dim, projector_hidden, bias=False, batchnorm_last=False,
                                     use_batchnorm=True)
            print(f"The projector has {projector_hidden} hidden units")
        else:
            self.projector = nn.Identity()
            print("No projector is used.")

    def initialize_loss(self, loss, temperature, n_mc):
        if loss == "NT-Xent":
            self.loss_fn = NTXent(temperature)
        elif "MCNT-Xent" in loss:
            self.loss_fn = MCNTXent(loss, temperature, n_mc)
        elif loss == "KL_Loss":
            self.loss_fn = KL_Loss(self.distribution_type, temperature)
        else:
            raise ValueError("Specify a correct loss.")

        print(f"We use the {loss}. Temperature set to {temperature}")

    def forward(self, x1, x2, epoch):
        dist1 = self.backbone_net(x1)
        dist2 = self.backbone_net(x2)

        var_reg = self.regularizer((dist1, dist2))
        ssl_loss = self.compute_ssl_loss(dist1, dist2, epoch)
        unc_loss = self.compute_uncertainty_loss(dist1, dist2)

        return ssl_loss, var_reg, unc_loss, (dist1, dist2)

    def compute_ssl_loss(self, dist1, dist2, epoch):
        n_batch = dist1.loc.shape[0]
        if self.loss == "NT-Xent":

            # Use distribution's location for the first epochs
            if epoch < self.loc_warmup:
                z1, z2 = dist1.loc, dist2.loc
            else:
                z1, z2 = dist1.rsample(), dist2.rsample()

            p1 = self.projector(z1)
            p2 = self.projector(z2)
            return self.loss_fn(p1, p2)

        if "MCNT-Xent" in self.loss:
            z1 = dist1.rsample((self.n_mc,)).view(n_batch * self.n_mc, -1)
            z2 = dist2.rsample((self.n_mc,)).view(n_batch * self.n_mc, -1)

            p1 = self.projector(z1)
            p2 = self.projector(z2)
            return self.loss_fn(p1.view(self.n_mc, n_batch, -1), p2.view(self.n_mc, n_batch, -1))

        if self.loss == "KL_Loss":
            if not self.projector_hidden:
                return self.loss_fn(dist1, dist2)
            elif self.projector_hidden and (self.distribution_type == "normal"):
                pz1, pk1 = self.projector(dist1.loc, dist1.scale)
                pz2, pk2 = self.projector(dist2.loc, dist2.scale)
                return self.loss_fn(Normal(pz1, pk1), Normal(pz2, pk2))
            else:
                return self.loss_fn(dist1, dist2)
        else:
            raise TypeError("Please specify a correct loss.")

    def compute_uncertainty_loss(self, dist1, dist2):
        if self.lambda_unc != 0.:
            unc_loss = self.uncertainty_loss(dist1, dist2)
        else:
            unc_loss = torch.tensor([0.0], device=dist1.loc.device)

        return unc_loss
