import torch
import torch.nn as nn

class UncertaintyLoss(nn.Module):

    def __init__(self, lambda_unc=0.):
        super().__init__()
        self.device = self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lambda_unc = lambda_unc

        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, dist1, dist2):
        z1, z2 = dist1.loc.detach(), dist2.loc.detach()

        bz = z1.shape[0]
        unc = torch.cat([dist1.scale, dist2.scale], dim=0)
        unc = torch.mean(unc.view(bz, -1), dim=1).to(self.device)

        sim = self.sim(z1, z1).to(self.device)
        loss = torch.mean((1 - sim) * unc - torch.log(unc))
        return loss * self.lambda_unc
