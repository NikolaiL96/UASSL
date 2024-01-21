from torch import nn
import torch
from distributions import powerspherical


class KL_Loss(nn.Module):

    def __init__(self, temperature):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature

    def kl_two_normals(self, mu1, mu2, var1, var2):
        mu1, mu2 = mu1.unsqueeze(1), mu2.unsqueeze(0)
        logvar1, logvar2 = torch.log(var1), torch.log(var2)
        logvar1, logvar2 = logvar1.unsqueeze(1), logvar2.unsqueeze(0)

        # KL Divergenz of two normal distributions with diagonal variance
        kld = 0.5 * torch.sum(logvar2 - logvar1 - 1 +
                              ((mu1 - mu2).pow(2) + logvar1.exp()).div(logvar2.exp()),
                              dim=-1)
        return kld

    def _mask(self, n_batch):
        mask_pos = torch.eye(2 * n_batch)
        mask_pos = mask_pos.roll(shifts=n_batch, dims=0)
        mask_pos = mask_pos.to(bool)
        return mask_pos.to(self.device)

    def forward(self, loc1, scale1, loc2, scale2):
        n_batch, rep_dim = loc1.shape
        loc = torch.cat([loc1, loc2], dim=0)

        if scale1.dim() == 1:
            scale = torch.cat([scale1, scale2], dim=0).unsqueeze(1).repeat(1, rep_dim)
        else:
            scale = torch.cat([scale1, scale2], dim=0)

        sim_mat = self.kl_two_normals(loc, loc, scale, scale)
        mask_pos = self._mask(n_batch)
        sim_mat = sim_mat.fill_diagonal_(-9e15)

        sim_mat = sim_mat / self.temperature
        loss = torch.logsumexp(sim_mat, dim=-1) - sim_mat[mask_pos]

        return torch.mean(loss)


class KL_PS_Loss(nn.Module):

    def __init__(self, temperature):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature

    def kl_ps_ps_mc(self, p, q, n_mc=1000):
        H_p = p.entropy()

        mc = p.rsample((n_mc,))  # [n_mc, n_batch, n_feats]
        m = torch.einsum("ib, njb -> nij", q.loc, mc)  # [n_mc, n_batch, n_batch]

        E_q = q.log_normalizer() + q.scale * torch.mean(torch.log(1 + m), dim=0)
        kl = - H_p[None] - E_q  # [n_batch, n_batch]
        return kl

    def _mask(self, n_batch):
        mask_pos = torch.eye(2 * n_batch)
        mask_pos = mask_pos.roll(shifts=n_batch, dims=0)
        mask_pos = mask_pos.to(bool)
        return mask_pos.to(self.device)

    def forward(self, p1, p2):
        n_batch = p1.loc.shape[0]

        loc = torch.cat([p1.loc, p2.loc], dim=0)
        scale = torch.cat([p1.scale, p2.scale], dim=0)
        p = q = powerspherical.PowerSpherical(loc, scale)

        sim_mat = self.kl_ps_ps_mc(p, q)
        mask_pos = self._mask(n_batch)
        sim_mat = sim_mat.fill_diagonal_(-9e15)

        sim_mat = sim_mat * self.temperature
        loss = sim_mat[mask_pos] - torch.logsumexp(sim_mat, dim=-1)

        return torch.mean(loss)
