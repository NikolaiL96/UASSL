import torch


def get_checkpoint_path(method, distribution):
    checkpoint_path = None
    if torch.cuda.is_available():
        cluster = True
    else:
        cluster = False

    if method == "SimCLR":
        if distribution.lower() == "powerspherical":
            if cluster:
                checkpoint_path = "/home/lorenzni/checkpoints/SimCLR_Powerspherical/epoch_1000.tar"
            else:
                checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/SimCLR/Powerspherical/epoch_1000.tar"

        else:
            if cluster:
                checkpoint_path = "/home/lorenzni/checkpoints/SimCLR_sphere/epoch_1000.tar"
            else:
                checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/BarlowTwins/epoch_1000.tar"
    elif method == "BarlowTwins":
        if cluster:
            checkpoint_path = "/home/lorenzni/checkpoints/BT_sphere/epoch_1000.tar"
        else:
            checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/SimCLR/epoch_1000.tar"

    return checkpoint_path


def clean_params(params):
    #del params["backbone_net.fc.layer.weight"]
    #del params["backbone_net.fc.layer.bias"]
    del params["projector.mlp.6.weight"]
    return params
