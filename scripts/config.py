import os
from utils.model_factory import MODEL_CONFIG, METHOD_CLS, DEFAULT_OPTIONS

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("VariationalSSL", interactive=True)

@ex.named_config
def BarlowTwins():
    method = "BarlowTwins"
    augmentation_type = "BYOL"
    method_params = MODEL_CONFIG[method][DEFAULT_OPTIONS]
    method_cls = MODEL_CONFIG[method][METHOD_CLS]

@ex.named_config
def SimCLR():
    method = "SimCLR"
    augmentation_type = "BYOL"
    method_params = MODEL_CONFIG[method][DEFAULT_OPTIONS]
    method_cls = MODEL_CONFIG[method][METHOD_CLS]

@ex.named_config
def Supervised():
    method = "Supervised"
    augmentation_type = "Supervised"
    method_params = MODEL_CONFIG[method][DEFAULT_OPTIONS]
    method_cls = MODEL_CONFIG[method][METHOD_CLS]

@ex.config
def my_config():
    method = ""
    seed = 3407
    dataset = 'cifar10'
    network = 'resnet18'
    eta = 0
    optim_params = None
    clip = 0.
    clip_type = "Norm"
    reduced_lr = False
    loss = None
    lambda_reg = 0.01
    temperature = 0.05
    n_mc = 16
    name = ""
    slug = "",
    fine_tuned = False,
    pretrained = False
    lambda_unc = 0.
    evaluate = False

    path = "./saved_runs/"
    artifact_dir = f'{path}/{name}'  # name
    batch_size = 1024
    epochs = 800
    warmup = 10
    data_root = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Code/SSL_VAE/data"

    # Enable/Disable tensorboard logging
    tb_logging = True
    fine_tune = False
    last_layer_relu = True

    train_params = {
        "num_epochs": int(epochs),
        "optimizer": SGD,
        "scheduler": CosineAnnealingLR,
        "warmup_epochs": int(warmup),
        "iter_scheduler": True,
        "evaluate_at": [10, 25, 50, 75] + list(range(100, int(epochs) + 1, 50)),
        "reduced_lr": reduced_lr,
    }

    ex.observers = [FileStorageObserver(artifact_dir)]

    dl_kwargs = {"batch_size": batch_size, "shuffle": True,
                 "num_workers": min(os.cpu_count(), 8)}

    eval_optim_params = {"lr": 0.2, "momentum": 0.9, "weight_decay": 0.}
    eval_params = {"optim_params": eval_optim_params, "num_epochs": 100}
    distribution_params = {"type": "powerspherical"}

