import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyLoss(nn.Module):

    def __init__(self, lambda_unc=0.):
        super().__init__()

        self.lambda_unc = lambda_unc

    def forward(self, dist1, dist2):
        z1, z2 = dist1.loc.detach(), dist2.loc.detach()

        n_batch = z1.shape[0]

        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

        sim = torch.diag(torch.matmul(z1, z2.T))

        unc = torch.cat([dist1.scale, dist2.scale], dim=0)
        unc = torch.mean(unc.view(n_batch, -1), dim=1).to(z1.device)

        loss = torch.mean((1 - sim) * unc - torch.log(unc))
        return loss * self.lambda_unc
