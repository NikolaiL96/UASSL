from torch import nn
import torch
import torch.nn.functional as F


class NTXent(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()

        self.temperature = temperature

    def mask(self, n_batch, device):
        mask_self = torch.eye(2 * n_batch, dtype=torch.bool, device=device)
        mask_pos = mask_self.roll(shifts=n_batch, dims=1)
        return mask_pos, mask_self

    def forward(self, p1, p2):
        n_batch, _ = p1.shape

        p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

        z = torch.cat([p1, p2], dim=0)
        sim_mat = torch.matmul(z, z.transpose(-2, -1)) / self.temperature

        mask_pos, mask_self = self.mask(n_batch, device=p1.device)
        sim_mat[mask_self] = float('-inf')

        loss = torch.logsumexp(sim_mat, dim=-1) - sim_mat[mask_pos]
        return loss.mean()
