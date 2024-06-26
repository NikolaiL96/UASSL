import torch
from torch import nn
import torch.nn.functional as F

from .utils import get_configuration

class MCNTXent(nn.Module):
    def __init__(self, loss: str, temperature: float = 0.1, n_mc: int = 16):
        super().__init__()

        self.temperature = temperature
        self.n_mc = n_mc

        self.method, self.reduction = get_configuration(loss)

    def mask(self, n_batch, n_mc, device):
        b = 2 * n_batch

        if self.method == "local":
            mask_self = torch.eye(b, dtype=torch.bool, device=device)
            mask_pos = mask_self.roll(shifts=n_batch, dims=1)

            mask_pos = mask_pos.unsqueeze(0).expand(n_mc, -1, -1)
            mask_self = mask_self.unsqueeze(0).expand(n_mc, -1, -1)

            return mask_self, mask_pos

        elif self.method == "pairwise":
            s = b * n_mc
            mask_self = torch.zeros([s, s], dtype=torch.bool, device=device)
            mask_pos = torch.zeros([s, s], dtype=torch.bool, device=device)

            for i in range(0, n_mc):
                mask_self += torch.diag(torch.ones(b * (n_mc - i), dtype=torch.bool, device=device), i * b)
                mask_pos += torch.diag(torch.ones(b * (n_mc - i) - n_batch, dtype=torch.bool, device=device), i * b + n_batch)

            mask_self = mask_self + mask_self.T
            mask_pos = mask_pos + mask_pos.T
            mask_neg = mask_pos + mask_self

            return mask_self, mask_pos, ~mask_neg


    def forward(self, p1, p2):
        n_mc, n_batch, _ = p1.shape

        p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

        z = torch.cat([p1, p2], dim=1)

        if self.method == "local":
            mask_self, mask_pos = self.mask(n_batch, n_mc, device=p1.device)

            sim_mat = torch.bmm(z, z.permute(0, 2, 1))
            sim_mat[mask_self] = float('-inf')
            sim_mat /= self.temperature

            pos = sim_mat[mask_pos].view(n_mc, 2 * n_batch)
            loss = pos - torch.logsumexp(sim_mat, dim=-1)

        if self.method == "pairwise":
            mask_self, mask_pos, mask_neg = self.mask(n_batch, n_mc, p1.device)
            z = z.view(n_mc * 2 * n_batch, -1)

            sim_mat = z.matmul(z.transpose(-2, -1))  # [2 * n_mc * n_batch, 2 * n_mc * n_batch]
            sim_mat[mask_self] = float('-inf')
            sim_mat /= self.temperature

            # Shape of the terms: [1, 2 * n_mc * n_batch], [n_mc, 2 * n_mc * n_batch]
            loss = sim_mat[mask_pos].view(-1, n_mc * 2 * n_batch) - torch.logsumexp(sim_mat, dim=-1)[None, :]

        if self.reduction == "mean":
            loss = torch.logsumexp(loss, dim=0) - torch.log(torch.ones(1, device=p1.device) * n_mc)
            return -loss.mean()

        elif self.reduction == "min":
            loss, _ = torch.exp(loss).min(dim=0)
            return -torch.log(loss).mean()

        elif self.reduction == "max":
            loss, _ = torch.exp(loss).max(dim=0)
            return -torch.log(loss).mean()

