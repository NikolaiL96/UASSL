from typing import Union

import torch
import torch.nn as nn
from distributions import Probabilistic_Layer, Probabilistic_Regularizer, Normal
from .utils import MLP
from losses import MCNTXent, NTXent, KL_Loss, UncertaintyLoss


class SimCLR(nn.Module):
    # @ex.capture
    def __init__(
            self,
            backbone_net: nn.Module,
            projector_hidden: Union[int, tuple] = (2048, 2048, 256),
            loss: str = "InfoNCE",
            temperature: float = 0.05,
            distribution_type: str = None,
            lambda_reg: float = 1.0,
            unc_loss: bool = False,
            lambda_unc: float = 0.001,
            n_mc: int = 16
    ):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.backbone_net = backbone_net
        self.rep_dim = self.backbone_net.fc.in_features
        self.backbone_net.fc = Probabilistic_Layer(distribution_type, in_features=self.rep_dim)
        self.projector_hidden = projector_hidden

        if projector_hidden:
            self.projector = MLP(self.rep_dim, projector_hidden, batchnorm_last=True, bias=False)
        else:  # Use no projector
            self.projector = nn.Identity()

        self.loss = loss
        self.unc_loss = unc_loss

        self.temperature = temperature
        self.n_mc = n_mc
        self.distribution_type = distribution_type

        # Verbose
        print(f"We initialize SimCLR with {self.rep_dim} dimensions and a {distribution_type} distribution.")

        if projector_hidden:
            print(f"The projector has {projector_hidden} hidden units")
        else:
            print(f"No projector is used.")

        if self.loss == "NT-Xent":
            self.loss_fn = NTXent(temperature)
        elif "MCNT-Xent" in self.loss:
            self.loss_fn = MCNTXent(loss, temperature, n_mc)
        elif self.loss == "KL_Loss":
            self.loss_fn = KL_Loss(self.distribution_type, temperature)

        print(f"We use the {self.loss}. Temperature set to {temperature}")

        if self.unc_loss:
            self.uncertainty_loss = UncertaintyLoss(lambda_unc)
            print("We use the Uncertainty Loss")

        # Regularizer for the generated distribution
        self.regularizer = Probabilistic_Regularizer(distribution_type, lambda_reg)

    def forward(self, x1, x2):
        dist1, dist2 = self.backbone_net(x1), self.backbone_net(x2)

        n_batch = dist1.loc.shape[0]
        SimCLR.kappa = torch.mean(torch.cat([dist1.scale, dist2.scale], dim=0), dim=-1)

        if self.loss == "NT-Xent":
            p1 = self.projector(dist1.loc)
            p2 = self.projector(dist2.loc)
            ssl_loss = self.loss_fn(p1, p2)

        elif "MCNT-Xent" in self.loss:
            z1 = dist1.rsample((self.n_mc,)).view(n_batch * self.n_mc, -1)
            z2 = dist2.rsample((self.n_mc,)).view(n_batch * self.n_mc, -1)

            p1 = self.projector(z1)
            p2 = self.projector(z2)
            ssl_loss = self.loss_fn(p1.view(self.n_mc, n_batch, -1), p2.view(self.n_mc, n_batch, -1))

        elif self.loss == "KL_Loss":
            if not self.projector_hidden:
                ssl_loss = self.loss_fn(dist1, dist2)
            elif self.projector and self.distribution_type == "normal":
                pz1, pk1 = self.projector(dist1.loc, dist1.std)
                pz2, pk2 = self.projector(dist2.loc, dist2.std)
                ssl_loss = self.loss_fn(Normal(pz1, pk1), Normal(pz2, pk2))
            else:
                raise TypeError("Please use the Probabilistic Projection head with Normal distribution.")

        var_reg = self.regularizer((dist1, dist2))

        if self.unc_loss:
            unc_loss = self.uncertainty_loss(dist1, dist2)
        else:
            unc_loss = torch.zeros(1, device=self.device)

        return ssl_loss, var_reg, unc_loss
