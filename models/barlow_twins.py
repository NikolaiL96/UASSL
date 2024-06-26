from typing import Union

import torch
import torch.nn as nn

from distributions import Probabilistic_Layer, Probabilistic_Regularizer

from .utils import MLP
from losses import BT_Loss, MCBT_Loss, UncertaintyLoss


class BarlowTwins(nn.Module):
    # @ex.capture
    def __init__(
            self,
            backbone_net: nn.Module,
            projector_hidden: Union[int, tuple] = (8192, 8192, 8192),
            lambda_bt: float = 0.0051,
            lambda_reg: float = 0,
            distribution_type: str = "powerspherical",
            loss: str = "BT_Loss",
            lambda_unc: float = 0.001,
            loc_warmup: int = 0,
            n_mc: int = 16,
    ):
        super().__init__()

        self.backbone_net = backbone_net
        self.rep_dim = self.backbone_net.fc.in_features
        self.loc_warmup = loc_warmup
        self.n_mc = n_mc

        if backbone_net.name != "UncertaintyNet":
            backbone_net.fc = nn.Identity()
            self.backbone_net.fc = Probabilistic_Layer(distribution_type, in_features=self.rep_dim)

        self.distribution_type = distribution_type
        self.lambda_unc = lambda_unc
        self.loss = loss

        # Define projector
        if projector_hidden:
            self.projector = MLP(self.rep_dim, projector_hidden, bias=False)
        else:  # Don't use projector
            self.projector = nn.Identity()

        if loss == "BT_Loss":
            self.loss_fn = BT_Loss(projector_hidden, self.rep_dim, lambda_bt)
            print(f"We use the {loss}.")
        elif loss == "MCBT_Loss":
            self.loss_fn = MCBT_Loss(projector_hidden, self.rep_dim, lambda_bt, n_mc)
        else:
            raise ValueError("Specify a correct loss.")

        # Verbose
        print(f"We initialize BarlowTwins with {self.rep_dim} dimensions and a {distribution_type} distribution")
        print(f"The projector has {projector_hidden} hidden units")

        if self.lambda_unc != 0.:
            self.uncertainty_loss = UncertaintyLoss(lambda_unc)
            print("We use the Uncertainty Loss")

        # Regularizer for the generated distribution
        self.regularizer = Probabilistic_Regularizer(distribution_type, lambda_reg)

    def forward(self, x1, x2, epoch):
        dist1 = self.backbone_net(x1)
        dist2 = self.backbone_net(x2)

        var_reg = self.regularizer((dist1, dist2))
        ssl_loss = self.compute_ssl_loss(dist1, dist2, epoch)
        unc_loss = self.compute_uncertainty_loss(dist1, dist2)

        return ssl_loss, var_reg, unc_loss, (dist1, dist2)

    def compute_ssl_loss(self, dist1, dist2, epoch):
        n_batch = dist1.loc.shape[0]

        if self.loss == "BT_Loss":
            # Use distribution's location for the first epochs
            if epoch < self.loc_warmup:
                z1, z2 = dist1.loc, dist2.loc
            else:
                z1, z2 = dist1.rsample(), dist2.rsample()

            p1 = self.projector(z1)
            p2 = self.projector(z2)
            return self.loss_fn(p1, p2)

        if "MCBT_Loss" in self.loss:
            z1 = dist1.rsample((self.n_mc,)).view(n_batch * self.n_mc, -1)
            z2 = dist2.rsample((self.n_mc,)).view(n_batch * self.n_mc, -1)

            p1 = self.projector(z1)
            p2 = self.projector(z2)
            return self.loss_fn(p1.view(self.n_mc, n_batch, -1), p2.view(self.n_mc, n_batch, -1))

    def compute_uncertainty_loss(self, dist1, dist2):
        if self.lambda_unc != 0.:
            unc_loss = self.uncertainty_loss(dist1, dist2)
        else:
            unc_loss = torch.tensor([0.0], device=dist1.loc.device)

        return unc_loss
