from torch import nn
import torch
import torch.nn.functional as F


class InfoNCE(nn.Module):

    def __init__(self, temperature: float = 0.1, normalize: bool = True):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.normalize = normalize
        self.temperature = temperature


    def _mask(self, n_batch):
        mask_pos = torch.eye(2 * n_batch)
        mask_pos = mask_pos.roll(shifts=n_batch, dims=0)
        mask_pos = mask_pos.to(bool)
        return mask_pos.to(self.device)

    def forward(self, p1, p2):
        n_batch, _ = p1.shape

        if self.normalize:
            p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

        embedding = torch.cat([p1, p2], dim=0)
        sim_mat = torch.matmul(embedding, embedding.T)

        mask_pos = self._mask(n_batch)
        sim_mat = sim_mat.fill_diagonal_(-9e15)

        sim_mat = sim_mat / self.temperature

        loss = torch.logsumexp(sim_mat, dim=-1) - sim_mat[mask_pos]
        return loss.mean()
