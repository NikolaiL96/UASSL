from torch import nn
import torch
import torch.nn.functional as F


class BT_Loss(nn.Module):

    def __init__(self, projector_hidden, rep_dim):
        super().__init__()

        self.bn = nn.BatchNorm1d(projector_hidden[-1] if projector_hidden else rep_dim, affine=False)

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2, lambda_bt):

        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))

        c_diff = (c - torch.eye(c.size(1)).to(z1.device)).pow(2)
        c_diff[~torch.eye(*c_diff.shape, dtype=torch.bool).to(z1.device)] *= lambda_bt
        return c_diff.sum()
