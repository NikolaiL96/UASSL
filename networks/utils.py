import torch


def get_checkpoint_path(method, distribution, model_options):
    try:
        lambda_reg, lambda_unc = model_options['lambda_reg'], model_options['lambda_unc']
    except KeyError:
        lambda_reg, lambda_unc = 0., 0.

    checkpoint_path = None
    if torch.cuda.is_available():
        cluster = True
    else:
        cluster = False

    if method == "SimCLR":
        if distribution.lower() == "powerspherical":
            if cluster:
                if lambda_reg > 0.:
                    checkpoint_path = "/home/lorenzni/checkpoints/SimCLR_PS_reg/epoch_150.tar"
                elif lambda_unc != 0.:
                    checkpoint_path = "/home/lorenzni/checkpoints/SimCLR_PS_unc/epoch_350.tar"
                else:
                    checkpoint_path = "/home/lorenzni/checkpoints/SimCLR_Powerspherical/epoch_1000.tar"
            else:
                checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/SimCLR/Powerspherical/epoch_1000.tar"

        else:
            if cluster:
                checkpoint_path = "/home/lorenzni/checkpoints/SimCLR_sphere/epoch_1000.tar"
            else:
                checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/BarlowTwins/epoch_1000.tar"

    elif method == "BarlowTwins":
        if distribution.lower() == "powerspherical":
            if cluster:
                if lambda_reg != 0.:
                    checkpoint_path = "/home/lorenzni/checkpoints/BT_PS_reg/epoch_1000.tar"
                elif lambda_unc != 0.:
                    checkpoint_path = "/home/lorenzni/checkpoints/BT_PS_unc/epoch_1000.tar"
                else:
                    checkpoint_path = "/home/lorenzni/checkpoints/BT_powerspherical/epoch_1000.tar"
        else:
            if cluster:
                checkpoint_path = "/home/lorenzni/checkpoints/BT_sphere/epoch_1000.tar"
            else:
                checkpoint_path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Checkpoints/SimCLR/epoch_1000.tar"

    elif method == "Supervised":
        if cluster:
            checkpoint_path = "/home/lorenzni/checkpoints/Supervised/epoch_200.tar"

    return checkpoint_path


def clean_params(params, distribution):

    if distribution.lower() in ["normal", "vonmisesfisher"]:
        del params["backbone_net.fc.layer.weight"]
        del params["backbone_net.fc.layer.bias"]

    try:
        del params["projector.mlp.6.weight"]
        return params
    except KeyError:
        return params
