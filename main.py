import os
from scripts.train import ex

import argparse
import sys
import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_projector_settings(method, projector):
    if projector:
        return (2048, 2048, 1024) if method == "BarlowTwins" else (2048, 2048, 256)
    return None

def get_data_root_and_path(cluster, run_final):
    if cluster:
        data_root = "./data/"
        path = "/home/lorenzni/runs_SSL_final" if run_final else "/home/lorenzni/runs_SSL"
    else:
        data_root = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Code/SSL_VAE/data"
        path = "./saved_runs/"
    return data_root, path

def get_optimizer(optimizer):
    if optimizer == "SGD":
        optim_params = {"lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4}
    if optimizer == "Lamb":
        optim_params = {"lr": args.lr, "max_grad_norm": 10., "weight_decay": 5e-4}
    return optim_params

if __name__ == "__main__":
    sys.path.append("")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", "-c", default=False, type=str2bool)
    parser.add_argument("--method", "-m", default="SimCLR")
    parser.add_argument("--epochs", "-e", default=800, type=int)
    parser.add_argument("--warmup", "-w", default=10, type=int)
    parser.add_argument("--distribution", "-dist", default="sphere")
    parser.add_argument("--dataset", "-d", default="cifar10")
    parser.add_argument("--lr", default=6e-2, type=float)
    parser.add_argument("--loss", "-l", default="NT-Xent")
    parser.add_argument("--lambda_reg", "-lam", default=0.001, type=float)
    parser.add_argument("--temperature", "-t", default=0.01, type=float)
    parser.add_argument("--batch_size", "-bs", default=64, type=int)
    parser.add_argument("--network", "-n", default="resnet18", type=str)
    parser.add_argument("--projector", "-pr", default=True, type=str2bool)
    parser.add_argument("--n_mc", default=16, type=int)
    parser.add_argument("--fine_tuned", default=False, type=str2bool)
    parser.add_argument("--lambda_bt", "-lbt", default=0.005, type=float)
    parser.add_argument("--lambda_unc", "-lu", default=0., type=float)
    parser.add_argument("--optimizer", default="SGD", type=str)
    parser.add_argument("--clip", default=0., type=float)
    parser.add_argument("--reduced_lr", default=False, type=str2bool)
    parser.add_argument("--run_final", "-rf", default=False, type=str2bool)

    args = parser.parse_args()

    data_root, path = get_data_root_and_path(args.cluster, args.run_final)
    projector = get_projector_settings(args.method, args.projector)

    if args.method == "SimCLR":
        method_params = {"projector_hidden": projector, "loss": args.loss, "lambda_reg": args.lambda_reg,
                         "temperature": args.temperature, "lambda_unc": args.lambda_unc,
                         "n_mc": args.n_mc}
    elif args.method == "BarlowTwins":
        method_params = {"projector_hidden": projector, "loss": args.loss, "lambda_bt": args.lambda_bt,
                         "lambda_reg": args.lambda_reg, "lambda_unc": args.lambda_unc}

    name = f"{args.method}--{args.dataset}--{args.network}--{projector}--{args.loss}"
    slug = f"{args.distribution}--t={args.temperature}--l_reg={args.lambda_reg}--l_unc={args.lambda_unc}"

    if args.cluster:
        slug += f"--ID:{os.getenv('SLURM_JOB_ID')}"

        if os.getenv('SLURM_JOB_NAME') != "gpu_job":
            slug += f"--Job_Name: {os.getenv('SLURM_JOB_NAME')}"

    time = datetime.datetime.now().strftime("%B%d")
    slug += f"--{time}"

    param_dict = {"name": name,
                  "slug": slug,
                  "network": args.network,
                  "method": args.method,
                  "dataset": args.dataset,
                  "optimizer": args.optimizer,
                  "optim_params": get_optimizer(args.optimizer),
                  "clip": args.clip,
                  "reduced_lr": args.reduced_lr,
                  "epochs": args.epochs,
                  "warmup": args.warmup,
                  "batch_size": args.batch_size,
                  "method_params": method_params,
                  "distribution_params": {"type": args.distribution},
                  "fine_tune": args.fine_tuned,
                  "lambda_unc": args.lambda_unc,
                  "path": path,
                  "data_root": data_root,
                  "n_mc": args.n_mc}

    ex.run(named_configs=[args.method], config_updates=param_dict)
