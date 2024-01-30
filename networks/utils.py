import torch


def get_checkpoint_path(method):
    checkpoint_path = None
    if torch.cuda.is_available():
        cluster = True
    else:
        cluster = False

    if method == "SimCLR":
        if cluster:
            checkpoint_path = "/home/lorenzni/runs_SSL_final/SimCLR--cifar10--resnet18--\(2048\,\ 2048\,\ 128\)--NT-Xent/sphere--t\=0.2--l_reg\=0.0--l_unc\=0.0--ID\:694501--Job_Name\:\ eta_reduced--January26/epoch_1000.tar"
        else:
            checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/BarlowTwins/epoch_1000.tar"
    elif method == "BarlowTwins":
        if cluster:
            checkpoint_path = "/home/lorenzni/runs_SSL_final/BarlowTwins--cifar10--resnet18--\(2048\,\ 2048\,\ 2048\)--BT_Loss/sphere--t\=0.01--l_reg\=0.0--l_unc\=0.0--ID\:694499--Job_Name\:\ eta_reduced--January26/epoch_1000.tar"
        else:
            checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/SimCLR/epoch_1000.tar"

    return checkpoint_path


def clean_params(params):
    del params["backbone_net.fc.layer.weight"]
    del params["backbone_net.fc.layer.bias"]
    del params["projector.mlp.6.weight"]
    return params
