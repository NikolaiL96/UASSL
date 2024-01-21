from typing import Union

import torch
import torch.nn as nn
from distributions import Probabilistic_Layer, Probabilistic_Regularizer
from .utils import MLP
from losses import MCInfoNCE, InfoNCE, KL_Loss, KL_PS_Loss, UncertaintyLoss


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
            lambda_unc: float = 0.001
    ):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.backbone_net = backbone_net
        self.rep_dim = self.backbone_net.fc.in_features
        self.backbone_net.fc = Probabilistic_Layer(distribution_type, in_features=self.rep_dim)

        if projector_hidden:
            self.projector = MLP(self.rep_dim, projector_hidden, batchnorm_last=True, bias=False)
        else:  # Use no projector
            self.projector = nn.Identity()

        self.loss = loss
        self.unc_loss = unc_loss
        self.distribution_type = distribution_type
        self.temperature = temperature
        self.projector_hidden = projector_hidden

        # Verbose
        print(f"We initialize SimCLR with {self.rep_dim} dimensions and a {distribution_type} distribution.")

        if projector_hidden:
            print(f"The projector has {projector_hidden} hidden units")
        else:
            print(f"No projector is used.")

        if self.loss == "InfoNCE":
            self.loss_fn = InfoNCE(temperature)
        elif self.loss == "KL_Loss":
            self.loss_fn = KL_Loss(temperature)
        elif self.loss == "KL_PS_Loss":
            self.loss_fn = KL_PS_Loss(temperature)

        print(f"We use the {self.loss}. Temperature set to {temperature}")

        if self.unc_loss:
            self.uncertainty_loss = UncertaintyLoss(lambda_unc)
            print("We use the Uncertainty Loss")

        # Regularizer for the generated distribution
        self.regularizer = Probabilistic_Regularizer(distribution_type, lambda_reg)

    def forward(self, x1, x2):
        # Get Distribution
        dist1, dist2 = self.backbone_net(x1), self.backbone_net(x2)

        SimCLR.kappa = torch.mean(torch.cat([dist1.scale, dist2.scale], dim=0), dim=-1)

        if self.loss == "InfoNCE":
            p1 = self.projector(dist1.loc)
            p2 = self.projector(dist2.loc)
            ssl_loss = self.loss_fn(p1, p2)

        elif self.loss == "KL_Loss":
            if self.projector_hidden:
                ssl_loss = self.loss_fn(dist1.loc, dist1.std, dist2.loc, dist2.std)
            else:
                pz1, pk1 = self.projector(dist1.loc, dist1.std)
                pz2, pk2 = self.projector(dist2.loc, dist2.std)
                ssl_loss = self.loss_fn(pz1, pk1, pz2, pk2)

        elif self.loss == "KL_PS_Loss":
            ssl_loss = self.loss_fn(dist1, dist2)

        var_reg = self.regularizer((dist1, dist2))

        if self.unc_loss:
            unc_loss = self.uncertainty_loss(dist1, dist2)
        else:
            unc_loss = torch.zeros(1, device=self.device)

        return ssl_loss, var_reg, unc_loss
