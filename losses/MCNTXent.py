from torch import nn
import torch
import torch.nn.functional as F

class MCNTXent(nn.Module):

    def __init__(self, temperature: float = 0.1, n_mc: int = 16):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.temperature = temperature
        self.n_mc = n_mc


    def _mask(self, n_batch, n_samples):
        B = 2 * n_batch
        if self.method == "simple":
            mask_pos = torch.eye(B)
            mask_pos = mask_pos.roll(shifts=n_batch, dims=0)
            mask_pos = mask_pos.to(bool)
            mask = mask_pos.unsqueeze(0).repeat(n_samples, 1, 1)

        elif self.method == "pairwise":
            for i in range(0, n_samples):
                h = B * n_samples - B * i
                m1 = torch.diag(torch.ones(h), diagonal=B * i)
                m2 = torch.diag(torch.ones(h - n_batch), diagonal=B * i + n_batch)
                try:
                    m = torch.add(m1, m2)
                    mask = torch.add(mask, m)
                except:
                    mask = torch.add(m1, m2)
            mask = torch.add(mask, torch.flip(mask, dims=[0, 1]))
            mask = mask.fill_diagonal_(0)
            mask = torch.gt(mask, 0).to(torch.bool)

        return mask.to(self.device)

    def forward(self, p1, p2, kappa: torch.tensor = None, t=None):
        B, _ = p1.shape
        n_batch = int(B / self.n_samples)
        p1, p2 = p1.view(self.n_samples, n_batch, -1), p2.view(self.n_samples, n_batch, -1)

        if self.normalize:
            p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)
        embedding = torch.cat([p1, p2], dim=1)

        if self.method == "simple":
            sim_mat = torch.bmm(embedding, embedding.permute(0, 2, 1))
            mask = self._mask(n_batch, self.n_samples)

            mask_diag = torch.eye(2*n_batch, dtype=torch.bool, device=self.device)
            mask_diag = mask_diag.unsqueeze(0).repeat(self.n_samples, 1, 1)

            sim_mat = sim_mat.masked_fill_(mask_diag, -9e15)

            if self.uncertainty_weighting == "temperature":
                t = t.view(1, 1, -1)
                sim_mat = sim_mat / t
            else:
                sim_mat = sim_mat / self.temperature

            positives = sim_mat[mask].view(self.n_samples, 2 * n_batch)

            if self.reduction == "min":
                min_positives, _ = torch.min(positives, dim=0)
                loss = torch.logsumexp(sim_mat, dim=-1) - min_positives[None, :]
            else:
                loss = torch.logsumexp(sim_mat, dim=-1) - positives
            loss = torch.logsumexp(loss, dim=0) - torch.log(torch.ones(1, device=self.device) * self.n_samples)
            loss = torch.mean(loss)

            if self.uncertainty_weighting == "weighted average":
                k_norm = F.softmax(kappa, dim=0).squeeze(1)
                loss = torch.sum(k_norm * loss)
            else:
                loss = loss.mean()

        elif self.method == "pairwise":
            embedding = embedding.view(self.n_samples * 2 * n_batch, -1)
            sim_mat = torch.matmul(embedding, embedding.T)
            mask_pos = self._mask(n_batch, self.n_samples)

            sim_mat = sim_mat.fill_diagonal_(-9e15)

            if self.uncertainty_weighting == "temperature":
                if self.uncertainty_weighting == "temperature" and kappa is None and self.method != "pairwise":
                    raise ValueError("Use pairwise method and/or provide kappa.")
                else:
                    k_norm = F.softmax(kappa, dim=0)
                    sim_mat = sim_mat / torch.repeat_interleave(k_norm.squeeze(1), self.n_samples)
            else:
                sim_mat = sim_mat / self.temperature

            positives = sim_mat[mask_pos].view(self.n_samples * 2*n_batch, -1)

            if self.reduction == "min":
                min_positives, _ = torch.min(positives, dim=-1)
                loss = torch.logsumexp(sim_mat, dim=-1) - min_positives
            else:
                loss = torch.logsumexp(sim_mat, dim=-1) - torch.logsumexp(positives, dim=-1)

            if self.uncertainty_weighting == "weighted average":
                k_norm = F.softmax(kappa, dim=0)
                loss = torch.repeat_interleave(k_norm.squeeze(1), self.n_samples) * loss
                loss = torch.sum(loss) / self.n_samples
            #elif self.uncertainty_weighting == "temperature":
                #max2 = torch.max(kappa)
                #kappa2 = torch.div(kappa, max2)
                #kappa3 = 1 - kappa2

            else:
                loss = loss.mean()

        return loss
