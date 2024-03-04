import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import os
from pathlib import Path

from matplotlib import rc
import matplotlib.pylab as plt

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def get_img(model, device, data):
    use_amp = device.type == 'cuda'

    uncertainty, labels, img = (), (), ()

    for n, (x, y) in enumerate(data.test_dl):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            with autocast(enabled=use_amp):
                dist = model.backbone_net(x)

            uncertainty += (dist.scale, )
            labels += (y, )
            img += (x, )

    uncertainty, labels, img = torch.cat(uncertainty), torch.cat(labels), torch.cat(img)

    idx_lower = torch.topk(-uncertainty, 2000).indices
    idx_upper = torch.topk(uncertainty, 2000).indices

    Y_low = labels[idx_lower]
    Y_high = labels[idx_upper]

    idx_class_high = []
    idx_class_low = []
    for i in range(10):
        y_i_low = Y_low == i
        y_i_high = Y_high == i
        idx_class_high.append(idx_upper[y_i_high.nonzero()[0]])
        idx_class_low.append(idx_lower[y_i_low.nonzero()[0]])

    idx_lower, idx_upper = torch.cat(idx_class_low), torch.cat(idx_class_high)

    low, high = img[idx_lower], img[idx_upper]
    uncertainty_low, uncertainty_high = uncertainty[idx_lower], uncertainty[idx_upper]

    low = inverse_normalize(low, device)
    high = inverse_normalize(high, device)

    low = F.interpolate(low, scale_factor=(4, 4), mode='bilinear')
    high = F.interpolate(high, scale_factor=(4, 4), mode='bilinear')

    return low, high, uncertainty_low, uncertainty_high

def plot_kappa_class(model, device, data):
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    X_lower, X_upper, kappa_lower, kappa_upper = get_img(model, device, data)
    kappa_lower = kappa_lower.type(torch.int64).cpu().numpy()
    kappa_upper = kappa_upper.type(torch.int64).cpu().numpy()
    X_lower = X_lower.permute(0, 2, 3, 1).cpu().numpy()
    X_upper = X_upper.permute(0, 2, 3, 1).cpu().numpy()

    fig, axs = plt.subplots(2, 10, figsize=(15, 3))
    for n, ax in enumerate(axs.reshape(-1)):
        if n <= 9:
            ax.imshow(X_upper[n], interpolation='nearest')
            ax.set_title(r"$\kappa=$ "f"{kappa_upper[n]}", fontsize=15)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 0:
                ax.set_ylabel("Most\nCertain", rotation='horizontal', ha='right', va='center', fontsize=15,
                              multialignment='center')


        else:
            ax.imshow(X_lower[n - 10], interpolation='nearest')
            ax.set_title(r"$\kappa=$ "f"{kappa_lower[n - 10]}", fontsize=15)
            ax.set_xlabel(f"{names[n - 10].title()}", fontsize=20)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 10:
                ax.set_ylabel("Least\nCertain", rotation='horizontal', ha='right', va='center', fontsize=15,
                              multialignment='center')
    fig.tight_layout()

    id = os.getenv('SLURM_JOB_ID')
    name = f"Uncertainty_CIFAR10"
    path = f"/home/lorenzni/imgs/{id}"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)


def inverse_normalize(tensor, device, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor
