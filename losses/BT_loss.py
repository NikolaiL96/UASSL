from torch import nn
import torch


class BT_Loss(nn.Module):

    def __init__(self, projector_hidden, rep_dim, lambda_bt):
        super().__init__()

        self.lambda_bt = lambda_bt
        self.bn = nn.BatchNorm1d(projector_hidden[-1] if projector_hidden else rep_dim, affine=False)

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))

        c_diff = (c - torch.eye(c.size(1)).to(z1.device)).pow(2)
        c_diff[~torch.eye(*c_diff.shape, dtype=torch.bool).to(z1.device)] *= self.lambda_bt
        return c_diff.sum()


class MCBT_Loss(nn.Module):

    def __init__(self, projector_hidden, rep_dim, lambda_bt, n_mc):
        super().__init__()

        self.lambda_bt = lambda_bt
        self.n_mc = n_mc
        self.bn = nn.BatchNorm1d(projector_hidden[-1] if projector_hidden else rep_dim, affine=False)

    def _bn_mc(self, z):
        n_mc, bs, proj = z.shape
        t_bn = torch.empty_like(z)

        for i in range(n_mc):
            sample = z[i]
            sample = sample.view(bs, proj)
            t_bn[i] = self.bn(sample).view(1, bs, proj)

        return t_bn

    def forward(self, z1, z2):
        _, bs, _ = z1.shape
        z1, z2 = self._bn_mc(z1), self._bn_mc(z2)

        c = torch.einsum("bij, bjk -> bik", z1.permute(0, 2, 1), z2)
        c.div_(z1.size(1))

        identity = torch.eye(c.size(1)).to(z1.device).expand(self.n_mc, -1, -1)

        # Calculate the squared difference from the identity matrix
        c_diff = (c - identity).pow(2)
        c_diff[~identity.to(torch.bool)] *= self.lambda_bt

        # Calculate the BarlowTwin loss separate for each MC sample
        c_diff = torch.sum(c_diff, dim=(1, 2))
        return c_diff.mean()
