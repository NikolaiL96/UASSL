from os import listdir
from os.path import isfile, join
from collections import OrderedDict
import logging

import requests
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.lamp import Lamb
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from distributions import powerspherical as ps


def log_level(log_level="info"):
    log_level = log_level.lower()

    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    return log_levels[log_level]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def get_projector_settings(method, projector, network, projector_out=None):
    p_dim = 512 if network == "resnet18" else 2048

    if (projector is True) and (projector_out is None):
        return (2048, 2048, 2048) if method == "BarlowTwins" else (p_dim, p_dim, 128)

    # We only want to have a custom projector out-dimension for BarlowTwins
    elif (projector is True) and (projector_out is not None):
        return (2048, 2048, projector_out) if method == "BarlowTwins" else (p_dim, p_dim, 128)

    else:
        return None


def get_data_root_and_path(cluster, run_final):
    if cluster:
        data_root = "./data/"
        path = "/home/lorenzni/runs_SSL_final" if run_final else "/home/lorenzni/runs_SSL"
    else:
        data_root = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Code/SSL_VAE/data"
        path = "./saved_runs/"
    return data_root, path


def get_optimizer(optimizer, method, lr=6e-2, batch_size=512):
    optim_params = {}

    if method == "SimCLR":
        optim_params["lr"] = 0.3 * batch_size / 256
        optim_params["weight_decay"] = 1.0e-6

    elif method == "BarlowTwins":
        optim_params["lr"] = 0.3 * batch_size / 512
        optim_params["weight_decay"] = 1.0e-4

    if optimizer == "SGD":
        optim_params["momentum"] = 0.9
    elif optimizer == "Lamb":
        optim_params["max_grad_norm"] = 10
    return optim_params


def get_train_params(method, optimizer, epochs, reduced_lr, batch_size, lr=6e-2):
    eta = 1.0e-6
    if method == "SimCLR":
        warmup = 0
    else:
        warmup = 10

    optim_params = get_optimizer(optimizer=optimizer, method=method, batch_size=batch_size, lr=lr)

    return {
               "num_epochs": int(epochs),
               "optimizer": SGD if optimizer == "SGD" else Lamb,
               "scheduler": CosineAnnealingLR,
               "warmup_epochs": int(warmup),
               "iter_scheduler": True,
               "evaluate_at": [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650, 700, 750, 800],
               "reduced_lr": reduced_lr,
               "optim_params": optim_params,
           }, eta


def check_existing_model(save_root, device, ask_user=False):
    """
    Epochs must be saved in format: f'epoch_{epoch:03}.tar'.
    E.g. for epoch 20: epoch_020.tar.
    """
    # Get all files

    # init
    epoch_start = 0
    saved_data = None
    if save_root:
        files = [f for f in listdir(save_root) if isfile(join(save_root, f))]
        if len(files) > 0:
            user_answer = "y"
            if ask_user:
                user_answer = "Users_answer"
                while user_answer not in ["y", "n"]:
                    user_answer = input("Pretrained model available, use it?[y/n]: ").lower()[0]
            if user_answer == "y":
                epoch_start = max([int(file[-7:-4]) for file in files])
                # Load data
                saved_data = torch.load(
                    join(save_root, f"epoch_{epoch_start:03}.tar"), map_location=device
                )

    return epoch_start, saved_data


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        info = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        info = "CPU"

    return device, info


def inverse_normalize(tensor, device, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201), ):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def _find_low_and_high_images(feature, target, uncertainty, loc, device, cl=0):
    idx_lower = torch.topk(-uncertainty, 1000).indices
    idx_upper = torch.topk(uncertainty, 1000).indices

    Y_low = target[idx_lower]
    Y_high = target[idx_upper]

    idx_class_high = []
    idx_class_low = []
    for i in range(10):
        y_i_low = Y_low == i
        y_i_high = Y_high == i
        idx_class_high.append(idx_upper[y_i_high.nonzero()[0]])
        idx_class_low.append(idx_lower[y_i_low.nonzero()[0]])

    low_indices, high_indices = torch.cat(idx_class_low), torch.cat(idx_class_high)
    low, high = feature[low_indices][cl].unsqueeze(0), feature[high_indices][cl].unsqueeze(0)
    kappa_low, kappa_high = uncertainty[low_indices][cl].unsqueeze(0), uncertainty[high_indices][cl].unsqueeze(0)
    loc_low, loc_high = loc[low_indices][cl].unsqueeze(0), loc[high_indices][cl].unsqueeze(0)

    dist_low = ps.PowerSpherical(loc_low, kappa_low)
    dist_high = ps.PowerSpherical(loc_high, kappa_high)

    low = inverse_normalize(low, device=device)
    high = inverse_normalize(high, device=device)

    low = F.interpolate(low, scale_factor=(4, 4), mode='bilinear')
    high = F.interpolate(high, scale_factor=(4, 4), mode='bilinear')

    low = low.permute(0, 2, 3, 1)
    high = high.permute(0, 2, 3, 1)
    return [dist_low, dist_high], [low, high]


def get_cifar10h():
    path = Path("./data/CIFAR_10H.csv")
    if path.is_file():
        data = np.genfromtxt(path, delimiter=",")
    else:
        r = requests.get(
            'http://raw.githubusercontent.com/jcpeterson/cifar-10h/48da312b193b82e6939ec20f23d1f09cbdd849c8/data/cifar10h-probs.npy')
        data = np.load(io.BytesIO(r.content))
        np.savetxt(path, data, delimiter=",")
    return data


def knn_predict(
        feature: torch.Tensor,
        feature_bank: torch.Tensor,
        feature_labels: torch.Tensor,
        num_classes: int,
        knn_k: int,
        knn_t: float,
) -> torch.Tensor:
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


@torch.no_grad()
def _get_projector(network_id, rep_dim, method=None, projector_out=None):
    if network_id in ["MnistResNet18", "MnistResNet34", "resnet10", "resnet18"]:
        if rep_dim == 3:
            projector = (3, 3, 3)
            out_dim = 3
        elif method == "BarlowTwin" and projector_out is None:
            projector = (2048, 2048, 1024)
        elif method == "BarlowTwin" and projector_out is not None:
            projector = (2048, 2048, projector_out)
        else:
            projector = (512, 512, 128)
            out_dim = None
    elif network_id in ["MnistResNet50", "resnet50"]:
        if rep_dim == 3:
            projector = (3, 3, 3)
            out_dim = 3
        else:
            projector = (2048, 2048, 128)
            out_dim = None
    elif rep_dim == 3:
        projector = (3, 3, 3)
        out_dim = None
    elif rep_dim <= 84:
        projector = (84, 84, 48)
        out_dim = None
    elif rep_dim <= 128:
        projector = (128, 128, 96)
        out_dim = None
    elif rep_dim <= 2048:
        projector = (2048, 2048, 128)
        out_dim = None
    return projector, out_dim


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


def _get_model_name(checkpoint_path):
    return [s for s in checkpoint_path.split("/") if any(s.startswith(p) for p in ["BarlowTwin", "SimCLR"])][0]


def _split_name(name):
    name = _get_model_name(name)
    split = name.split("--")

    network_id = split[1]
    dataset = split[2]
    distribution = split[3]
    lambda_reg = float(split[4].split("=")[1])
    rep_dim = int(split[5])
    projector, out_dim = _get_projector(network_id, rep_dim=rep_dim)

    if split[0] == "BarlowTwin":
        model_id = "barlowtwin"
        mc = str_to_bool(split[6].split("=")[1])
        lambda_bt = float(split[7].split("=")[1])
        if mc:
            n_samples = int(split[8])
        else:
            n_samples = "None"

        method_params = {"projector_hidden": projector, "lambd": lambda_bt, 'mc': mc,
                         "lambda_reg": lambda_reg,
                         "n_samples": n_samples}
    elif split[0] == "SimCLR":
        model_id = "simclr"
        loss = split[6]
        temperature = float(split[7])
        uncertainty_weighting = split[8]
        if loss == "MCInfoNCE":
            n_samples = int(split[9].split("=")[1])
            loss_method = split[10]
            reduction = split[11]

            method_params = {"loss": loss, "projector_hidden": projector,
                             "loss_method": loss_method, "temperature": temperature,
                             "lambda_reg": lambda_reg, "n_samples": n_samples,
                             "reduction": reduction,
                             "uncertainty_weighting": uncertainty_weighting, "rep_dim": out_dim}
        else:
            method_params = {"projector_hidden": projector, "loss": loss, "lambda_reg": lambda_reg,
                             "temperature": temperature, "uncertainty_weighting": uncertainty_weighting,
                             "n_samples": "None"}

    model_params = {"model_id": model_id,
                    "network_id": network_id,
                    "rep_dim": rep_dim,
                    "distribution_type": distribution,
                    "model_options": method_params}

    return model_params, dataset


def print_parameter_status(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            status = "Not Frozen"
        else:
            status = "Frozen"
        print(f"Layer: {name}, Parameter Status: {status}")


def _get_dict(checkpoint, starts_with):
    _state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace(starts_with, "")
        _state_dict[name] = v
    return _state_dict


def _get_state_dict(checkpoint):
    state_dict_projector = OrderedDict((key, checkpoint[key]) for key in checkpoint if key.startswith("proj"))
    state_dict_backbone = OrderedDict((key, checkpoint[key]) for key in checkpoint if key.startswith("backbone"))

    _state_dict_backbone = _get_dict(state_dict_backbone, "backbone_net.")
    _state_dict_projector = _get_dict(state_dict_projector, "projector.")

    return _state_dict_backbone, _state_dict_projector
