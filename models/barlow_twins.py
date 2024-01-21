from typing import Union

import torch
import torch.nn as nn

from distributions import Probabilistic_Layer, Probabilistic_Regularizer

from .utils import MLP
from losses import BT_Loss, UncertaintyLoss


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
            lambda_unc: float = 0.001
    ):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.backbone_net = backbone_net
        self.rep_dim = self.backbone_net.fc.in_features

        if backbone_net.name != "UncertaintyNet":
            self.backbone_net.fc = Probabilistic_Layer(distribution_type, in_features=self.rep_dim)

        self.distribution_type = distribution_type

        self.lambda_bt = lambda_bt
        self.lambda_unc = lambda_unc
        self.backbone_net = backbone_net

        # Define projector
        if projector_hidden:
            self.projector = MLP(self.rep_dim, projector_hidden, bias=False, batchnorm_last=True)
        else:  # Don't use projector
            self.projector = nn.Identity()

        if loss == "BT_Loss":
            self.loss_fn = BT_Loss(projector_hidden, self.rep_dim)
            print(f"We use the {loss}.")
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

    def forward(self, x1, x2):
        # Get Distribution
        dist1 = self.backbone_net(x1)
        dist2 = self.backbone_net(x2)

        BarlowTwins.kappa = torch.mean(torch.cat([dist1.scale, dist2.scale], dim=0), dim=-1)

        # Get Sample Projections
        p1 = self.projector(dist1.rsample())
        p2 = self.projector(dist2.rsample())

        # Get standard barlow twin loss
        ssl_loss = self.loss_fn(p1, p2, self.lambda_bt)
        var_reg = self.regularizer((dist1, dist2))

        if self.lambda_unc != 0.:
            unc_loss = self.uncertainty_loss(dist1, dist2)
        else:
            unc_loss = torch.zeros(1, device=self.device)

        return ssl_loss, var_reg, unc_loss
