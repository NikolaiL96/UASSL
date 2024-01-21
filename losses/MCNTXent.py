import torch
from torch import nn
import torch.nn.functional as F

from .utils import get_configuration

class MCNTXent(nn.Module):
    def __init__(self, loss: str, temperature: float = 0.1, n_mc: int = 16):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.n_mc = n_mc

        self.method, self.reduction = get_configuration(loss)

    def mask(self, n_batch, n_mc):
        B = 2 * n_batch
        if self.method == "local":
            mask_pos = torch.eye(B)
            mask_pos = mask_pos.roll(shifts=n_batch, dims=0)
            mask_pos = mask_pos.to(bool)
            mask_pos = mask_pos.unsqueeze(0).repeat(n_mc, 1, 1)

            mask_self = torch.eye(2 * n_batch, dtype=torch.bool, device=self.device)
            mask_self = mask_self.unsqueeze(0).repeat(n_mc, 1, 1)

            return mask_self, mask_pos

        elif self.method == "pairwise":
            b = 2 * n_batch
            s = b * n_mc
            mask_self = torch.zeros([s, s])
            mask_pos = torch.zeros([s, s])

            for i in range(0, n_mc):
                mask_self += torch.diag(torch.ones(b * (n_mc - i)), i * b)
                mask_pos += torch.diag(torch.ones(b * (n_mc - i) - n_batch), i * b + n_batch)

            mask_self = mask_self + mask_self.T
            mask_pos = mask_pos + mask_pos.T
            mask_neg = mask_pos + mask_self

            return mask_self.to(bool), mask_pos.to(bool), ~mask_neg.to(bool)

    def forward(self, p1, p2):
        n_mc, n_batch, _ = p1.shape
        z = torch.cat([p1, p2], dim=1)

        if self.method == "local":
            mask_self, mask_pos = self.mask(n_batch, n_mc)

            sim_mat = torch.bmm(z, z.permute(0, 2, 1))
            sim_mat = sim_mat.masked_fill_(mask_self, -9e15) / self.temperature

            pos = sim_mat[mask_pos].view(n_mc, 2 * n_batch)

            if self.reduction == "mean":
                l1 = torch.logsumexp(sim_mat, dim=-1)
                l2 = pos
                loss = l1 - l2
                loss = loss.mean()
                return loss

            elif self.reduction == "min":
                pass

        if self.method == "pairwise":
            mask_self, mask_pos, mask_neg = self.mask(n_batch, n_mc)
            z = z.view(n_mc * 2 * n_batch, -1)

            sim_mat = z @ z.T
            sim_mat.masked_fill_(mask_self, -9e15) / self.temperature

        #
        #
        #     if self.reduction == "min":
        #         min_positives, _ = torch.min(positives, dim=0)
        #         loss = torch.logsumexp(sim_mat, dim=-1) - min_positives[None, :]
        #     else:
        #         loss = torch.logsumexp(sim_mat, dim=-1) - positives
        #     loss = torch.logsumexp(loss, dim=0) - torch.log(torch.ones(1, device=self.device) * self.n_samples)
        #     loss = torch.mean(loss)
        #
        #     if self.uncertainty_weighting == "weighted average":
        #         k_norm = F.softmax(kappa, dim=0).squeeze(1)
        #         loss = torch.sum(k_norm * loss)
        #     else:
        #         loss = loss.mean()
        #
        # elif self.method == "pairwise":
        #     embedding = embedding.view(self.n_samples * 2 * n_batch, -1)
        #     sim_mat = torch.matmul(embedding, embedding.T)
        #     mask_pos = self._mask(n_batch, self.n_samples)
        #
        #     sim_mat = sim_mat.fill_diagonal_(-9e15)

        #
        #     positives = sim_mat[mask_pos].view(self.n_samples * 2 * n_batch, -1)
        #
        #     if self.reduction == "min":
        #         min_positives, _ = torch.min(positives, dim=-1)
        #         loss = torch.logsumexp(sim_mat, dim=-1) - min_positives
        #     else:
        #         loss = torch.logsumexp(sim_mat, dim=-1) - torch.logsumexp(positives, dim=-1)
        #
        #     if self.uncertainty_weighting == "weighted average":
        #         k_norm = F.softmax(kappa, dim=0)
        #         loss = torch.repeat_interleave(k_norm.squeeze(1), self.n_samples) * loss
        #         loss = torch.sum(loss) / self.n_samples
        #     # elif self.uncertainty_weighting == "temperature":
        #     # max2 = torch.max(kappa)
        #     # kappa2 = torch.div(kappa, max2)
        #     # kappa3 = 1 - kappa2
        #
        #     else:
        #         loss = loss.mean()
        #
        # return loss
