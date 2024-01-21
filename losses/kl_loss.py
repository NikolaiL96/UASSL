from torch import nn
import torch
from distributions import powerspherical


class KL_Loss(nn.Module):

    def __init__(self, distribution: str, temperature: float = 0.01):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.distribution = distribution

    def kl_two_powerspherical_mc(self, p, q, n_mc=1000):
        H_p = p.entropy()

        mc = p.rsample((n_mc,))  # [n_mc, n_batch, n_feats]
        m = torch.einsum("ib, njb -> nij", q.loc, mc)  # [n_mc, n_batch, n_batch]

        E_q = q.log_normalizer() + q.scale * torch.mean(torch.log1p(m), dim=0)
        kl = - H_p[None] - E_q  # [n_batch, n_batch]
        return kl

    def kl_two_ps_mc(self, p, q, n_mc=10000, chunk=10):
        # Version that calculates the log-term of the KL divergence in chunks for memory optimisation.
        H_p = p.entropy()

        log_term_chunks = []
        for i in range(0, n_mc, 10):
            mc = p.rsample((chunk,))  # [chunk_size, n_batch, n_feats]
            m = torch.einsum("ib, njb -> nij", q.loc, mc)  # [chunk_size, n_batch, n_batch]

            # Compute E_q for the chunk and store it
            log_term_chunk = torch.mean(torch.log1p(m), dim=0)
            log_term_chunks.append(log_term_chunk)

        log_term = torch.mean(torch.stack(log_term_chunks), dim=0)
        E_q = q.log_normalizer() + q.scale * log_term

        kl = - H_p[None] - E_q  # [n_batch, n_batch]
        return kl

    def kl_two_normals(self, mu1, mu2, var1, var2):

        if var1.dim() == 1:
            var1 = var1.unsqueeze(1).expand(1, mu1.shape[1])
            var2 = var2.unsqueeze(1).expand(1, mu1.shape[1])

        mu1, mu2 = mu1.unsqueeze(1), mu2.unsqueeze(0)
        logvar1, logvar2 = torch.log(var1).unsqueeze(1), torch.log(var2).unsqueeze(0)

        # KL Divergenz of two normal distributions with diagonal variance
        kld = 0.5 * torch.sum(logvar2 - logvar1 - 1 +
                              ((mu1 - mu2).pow(2) + logvar1.exp()).div(logvar2.exp()),
                              dim=-1)
        return kld

    def mask(self, n_batch):
        mask_self = torch.eye(2 * n_batch, dtype=torch.bool, device=self.device)
        mask_pos = mask_self.roll(shifts=n_batch, dims=1)
        return mask_pos, mask_self

    def forward(self, p1, p2):
        n_batch = p1.loc.shape[0]

        loc = torch.cat([p1.loc, p2.loc], dim=0)
        scale = torch.cat([p1.scale, p2.scale], dim=0)

        if "powerspherical" in self.distribution:
            p = q = powerspherical.PowerSpherical(loc, scale)
            sim_mat = self.kl_two_ps_mc(p, q)

        elif "normal" in self.distribution:
            sim_mat = self.kl_two_normals(loc, loc, scale, scale)

        mask_pos, mask_self = self.mask(n_batch)
        sim_mat[mask_self] = -9e6

        sim_mat *= self.temperature
        loss = sim_mat[mask_pos] - torch.logsumexp(sim_mat, dim=-1)

        return torch.mean(loss)
