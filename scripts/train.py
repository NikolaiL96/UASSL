import copy
import os
from scripts.config import ex
import datetime

import torch
import random
import numpy as np
from trainer import SSL_Trainer, SupervisedTrainer, Evaluator

from utils import get_device, load_dataset
from utils.model_factory import ModelFactory


@ex.capture
def _slug(dataset, method, distribution_params, method_params):
    time = datetime.datetime.now().strftime("%I:%M%p%B%d")
    return f"{dataset}-{method}--dist-{distribution_params['type']}--beta{distribution_params['beta']}" \
           f"--ID:{os.getenv('SLURM_JOB_ID')}--{time}"


@ex.automain
def main(
        artifact_dir,
        augmentation_type,
        data_root,
        distribution_params,
        dl_kwargs,
        method,
        method_cls,
        method_params,
        slug,
        lambda_reg,
        optim_params,
        evaluate,
        clip,
        clip_type,
        eta,
        train_params,
        eval_params,
        dataset,
        network,
        seed,
        fine_tune,
        pretrained,
        _run,
        _log
):
    save_root = f"{artifact_dir}/{slug}"
    _log.info(f"using data from `{data_root}`")
    _log.info(f"Result will be saved to {save_root}")

    device, device_info = get_device()
    _log.info(f"We are running on {device_info} (id={device}).")
    os.makedirs(save_root, exist_ok=True)

    # First set a Random Seed:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ssl_data, _, in_channels = load_dataset(dataset, data_root, augmentation_type, dl_kwargs)

    distribution_type = distribution_params["type"]

    params = {"model_id": method, "network_id": network, "model_options": method_params, "device": device,
              "in_channels": in_channels, "distribution_type": distribution_type, "pretrained": pretrained}

    model_factory = ModelFactory(**params)
    model = model_factory.build_model()
    save_root = f"{artifact_dir}/{slug}"
    model.to(device)

    if evaluate is True:
        Evaluator(model, ssl_data=ssl_data, eval_params=eval_params, device=device, distribution=distribution_type)
    elif method == "Supervised":
        cifar10_trainer = SupervisedTrainer(model, ssl_data=ssl_data, data_root=data_root, device=device,
                                            save_root=save_root,
                                            fine_tune=fine_tune, distribution=distribution_type, train_data=dataset,
                                            clip=clip,
                                            clip_type=clip_type)
    else:
        cifar10_trainer = SSL_Trainer(model, ssl_data=ssl_data, data_root=data_root, device=device, save_root=save_root,
                                      fine_tune=fine_tune, distribution=distribution_type, train_data=dataset,
                                      clip=clip,
                                      clip_type=clip_type)

    scheduler_params = {"T_max": (train_params["num_epochs"] - train_params["warmup_epochs"]) * len(ssl_data.train_dl),
                        "eta_min": eta}

    # Train
    if not evaluate:
        cifar10_trainer.train(**train_params, scheduler_params=scheduler_params, eval_params=eval_params)
