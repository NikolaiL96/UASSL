import numpy as np
from scipy.stats import entropy, spearmanr
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc as auc
import torchvision.transforms.functional as TF

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import os
import csv

from .model_factory import ModelFactory
from utils import load_dataset
from utils import Linear_Protocoler

from utils.utils import _find_low_and_high_images, get_cifar10h, knn_predict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


class Validate:

    def __init__(self, data, in_channel, checkpoint_path=None, model=None, epoch=None, last_epoch=False, name=None,
                 oob_data=None, oob_test=False, device=None, plot_tsne=False, main=False, train_data=None):

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.last_epoch = last_epoch

        self.oob_test = oob_test
        self.plot_tsne = plot_tsne
        self.oob_data = oob_data
        self.train_data = train_data
        self.main = main

        self.data = data

        self.dl_kwargs = {"batch_size": 512, "shuffle": True, "num_workers": min(os.cpu_count(), 0)}

        if model is not None:
            self.checkpoint_path = name
            self.model = model
            self.name = name
            self.model_params, dataset = self._split_name(self.name)

        elif checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            self.name = self._get_model_name(checkpoint_path)
            self.model_params, dataset = self._split_name(self.name)
            self.model = self._get_model(checkpoint_path, self.model_params, in_channel)
        else:
            raise ValueError("Pleas provide model path or model")

        if self.model_params['rep_dim'] == 3:
            self.plotting = True
        else:
            self.plotting = False

        if oob_data is not None:
            self.data_test, *_ = load_dataset(oob_data, "./data/",
                                              augmentation_type="BYOL",
                                              dl_kwargs=self.dl_kwargs)
            # self.data = self.data_test
        else:
            self.data_test = self.data

        self.encoder = self.model.backbone_net
        self.encoder.eval()

        if epoch is not None:
            self.epoch = epoch

        else:
            name_epoch = [s for s in checkpoint_path.split("/") if s.startswith("epoch")][0]
            self.epoch = name_epoch.split("epoch")[1].split(".")[0]
            self.last_epoch = True

        self.ID = self.model_params["ID"]

    @torch.no_grad()
    def _get_roc(self, loc, labels, kappa):
        with torch.no_grad():
            closest_idxes = loc.matmul(loc.transpose(-2, -1)).topk(2)[1][:, 1]
            closest_classes = labels[closest_idxes]
            is_same_class = (closest_classes == labels).float()
            auroc = auc(-kappa.squeeze(), is_same_class.int()).item()

        return is_same_class.mean(), auroc

    @torch.no_grad()
    def _get_cor(self):

        test_kappa = ()
        dl_kwargs = {"batch_size": 512, "shuffle": False, "num_workers": min(os.cpu_count(), 0)}
        data_cifar10h, *_ = load_dataset("cifar10", "./data/", augmentation_type="BYOL", dl_kwargs=dl_kwargs)

        for n, (x_test, labels_test) in enumerate(data_cifar10h.test_dl):
            with torch.no_grad():
                feats_test = self.encoder(x_test.to(self.device))
                kappa_test = feats_test.scale

            test_kappa += (kappa_test,)

        kappa = torch.cat(test_kappa).cpu().numpy()

        try:
            cifar10h = get_cifar10h()
            unc = 1 - entropy(cifar10h, axis=1)

            corr = np.corrcoef(kappa, unc)
            rank_corr = spearmanr(kappa, unc)

            return corr[0, 1], rank_corr.statistic

        except Exception as e:
            print(e)
            return torch.zeros(1), torch.zeros(1)

    def _get_linear_probing(self, epochs=10, lr=1e-2, oob_data=False):
        if not oob_data:
            test_loader = self.data.test_dl
        else:
            test_loader = self.data_test.test_dl
        linear_evaluator = Linear_Protocoler(self.model.backbone_net,
                                             repre_dim=self.model.repre_dim,
                                             variational=True,
                                             device=self.device)
        linear_evaluator.train(self.data.train_eval_dl, epochs, lr, None)
        return linear_evaluator.linear_accuracy(test_loader)

    @torch.no_grad()
    def _get_metrics(self, knn_k: int = 200, knn_t: float = 0.1, num_classes=None):

        train_features = ()
        train_labels = ()
        train_kappa = ()
        test_features = ()
        test_labels = ()
        test_kappa = ()
        test_loc = ()
        img = []

        Recall = []
        Auroc = []
        total_top1, total_num = 0.0, 0

        for n, (x_train, labels_train) in enumerate(self.data.train_eval_dl):
            with torch.no_grad():
                feats_train = self.encoder(x_train.to(self.device))
                kappa_train = feats_train.scale
                feats_train = feats_train.loc
            train_features += (F.normalize(feats_train, dim=1),)
            train_labels += (labels_train,)
            train_kappa += (kappa_train,)

            # if n == 1:
            #     break

        train_features = torch.cat(train_features)
        train_labels = torch.cat(train_labels).to(self.device)

        if num_classes is None:
            num_classes = len(set(train_labels.cpu().numpy().tolist()))

        for n, (x_test, labels_test) in enumerate(self.data_test.test_dl):
            with torch.no_grad():
                feats_test = self.encoder(x_test.to(self.device))
                labels_test = labels_test.to(self.device)
                kappa_test = feats_test.scale
                loc_test = feats_test.loc

                if self.model_params["distribution_type"] not in ["sphere", "normal"]:
                    kappa_test = 1 / kappa_test
                else:
                    kappa_test = kappa_test

                recall, auroc = self._get_roc(loc_test, labels_test, kappa_test)

                if not self.oob_test:
                    pred_labels = knn_predict(loc_test, train_features.t().contiguous(), train_labels, num_classes,
                                              knn_k,
                                              knn_t)
                    total_num += x_test.size(0)
                    total_top1 += (pred_labels[:, 0] == labels_test).float().sum().item()

            test_features += (F.normalize(loc_test, dim=1),)

            if self.plotting:
                img.append(x_test.cpu())
            test_labels += (labels_test.cpu(),)
            test_kappa += (kappa_test,)
            test_loc += (loc_test,)

            Recall.append(recall)
            Auroc.append(auroc)

            # if n == 0:
            #     break

        test_features = torch.cat(test_features)
        test_labels = torch.cat(test_labels).to(self.device)
        uncertainty = torch.cat(test_kappa).to(self.device)
        test_loc = torch.cat(test_loc).to(self.device)

        if self.plotting and self.last_epoch:
            img = torch.cat(img).to(self.device)
            kappa_idx, img_idx = _find_low_and_high_images(img.cpu(), test_labels.cpu(), uncertainty.cpu(),
                                                           test_loc.cpu(), device=torch.device('cpu'))
            self.plot_kappa(kappa_idx, img_idx)

        Recall = torch.stack(Recall, 0).mean()
        Auroc = torch.Tensor(Auroc).mean()

        if not self.oob_test:
            knn = torch.tensor(total_top1 / total_num * 100)
        else:
            knn = torch.zeros(1)

        if self.last_epoch:
            p_corrupted, cor_corrupted = self.corrupted_img()

            self._write_results(knn, Recall, Auroc)
        else:
            p_corrupted = cor_corrupted = torch.zeros(1)

        if self.plot_tsne:
            try:
                dataset = str(self.data_test.test_dl.dataset).split("\n")[0].split(" ")[1]
            except Exception as e:
                print(e, "Test-dataset could not be figured out from dataloader.")
                dataset = "Aux_Data"

            self.vis_t_SNE(test_features, test_labels, uncertainty, dataset)

        return Auroc, Recall, knn, cor_corrupted, p_corrupted

    @torch.no_grad()
    def corrupted_img(self):
        # Modified from https://github.com/mkirchhof/url/tree/main
        crop_min = torch.tensor(0.1, device=self.device)
        crop_max = torch.tensor(0.1, device=self.device)
        k = 0
        crop = torch.rand(len(self.data_test.test_dl.dataset), device=self.device) * (crop_max - crop_min) + crop_min

        Unc = ()
        Unc_c = ()

        for n, (x, target) in enumerate(self.data_test.test_dl):
            x, target = x.to(self.device), target.to(self.device)
            x_c = torch.zeros_like(x, device=self.device)
            c_sizes = torch.zeros(x.shape[0], device=self.device)
            for i in range(x.shape[0]):
                # Crop each image individually because torchvision cannot do it batch-wise
                crop_size = int(torch.round(min(x.shape[2], x.shape[3]) * crop[k + i]))
                c_sizes[i] = crop_size
                x_c[i] = TF.resize(TF.center_crop(x[i], [crop_size]), [x.shape[2], x.shape[3]], antialias=True)

            Unc += (self.encoder(x).scale,)
            Unc_c += (self.encoder(x_c.to(self.device)).scale,)

        Unc = 1 / torch.cat(Unc).to(self.device)
        Unc_c = 1 / torch.cat(Unc_c).to(self.device)

        p_cropped = (Unc < Unc_c).float().mean()
        cor_cropped = spearmanr(-Unc_c.cpu().numpy(), crop.cpu().numpy())[0]
        return p_cropped, cor_cropped

    @torch.no_grad()
    def get_zero_shot_metrics(self, dataset):
        if dataset == "sop":
            ssl_data, _, _ = load_dataset("sop", "/home/lorenzni/data/online_products/images/", "BYOL",
                                          dl_kwargs=self.dl_kwargs)
        elif dataset == "cup200":
            ssl_data, _, _ = load_dataset("cup200", "/home/lorenzni/data/cub200/images", "BYOL",
                                          dl_kwargs=self.dl_kwargs)
        elif dataset == "cars196":
            ssl_data, _, _ = load_dataset("cars196", "/home/lorenzni/data/cars196/images", "BYOL",
                                          dl_kwargs=self.dl_kwargs)

        Recall = []
        Auroc = []

        test_labels = ()
        test_kappa = ()
        test_loc = ()

        with torch.no_grad():
            for n, (x, labels) in enumerate(ssl_data.test_dl):
                feats = self.encoder(x.to(self.device))
                labels = labels.to(self.device)
                kappa = feats.scale
                loc = feats.loc

                if self.model_params["distribution_type"] not in ["sphere", "normal"]:
                    kappa = 1 / kappa
                else:
                    kappa = kappa

                recall, auroc = self._get_roc(loc, labels, kappa)

                test_labels += (labels.cpu(),)
                test_kappa += (kappa,)
                test_loc += (loc,)
                Recall.append(recall)
                Auroc.append(auroc)

                # if n == 0:
                #     break

            Labels = torch.cat(test_labels).to(self.device)
            U = torch.cat(test_kappa).to(self.device)
            Loc = torch.cat(test_loc).to(self.device)
            self.vis_t_SNE(Loc, Labels, U, dataset)

            Recall = torch.stack(Recall, 0).mean()
            Auroc = torch.Tensor(Auroc).mean()

        return Recall, Auroc

    def _write_results(self, knn, Recall, Auroc):
        if self.main:
            path = "/home/lorenzni/results.csv"
            params, _ = self._split_name(self.name)
        else:
            path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/results.csv"
            params = self.model_params

        if params['model_id'] == "barlowtwins":
            method = "-"
            if params['model_options']["mc"]:
                loss = "MC-BT-Loss"
                reduction = "mean"
            else:
                loss = "Standard-BT-Loss"
                reduction = "-"
        else:
            loss = params['model_options']['loss']
            if loss == "MCInfoNCE":
                reduction = params['model_options']['reduction']
                method = params['model_options']['loss_method']
            else:
                reduction = "-"
                method = "-"

        if not os.path.exists(path):
            with open(path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['Model', 'Network', 'ID', 'Distribution', 'Loss', 'Reduction', 'Method',
                                     'n Sample', 'Epochs', 'KNN', 'R@1', 'AUROC'])

        with open(path, 'a', newline='') as file:
            csv_writer = csv.writer(file)

            data_to_write = [params['model_id'], params["network_id"], params["ID"], params['distribution_type'], loss,
                             reduction, method, params["model_options"]['n_samples'], self.epoch, round(knn.item(), 2),
                             round(Recall.item(), 4), round(Auroc.item(), 4)]
            csv_writer.writerow(data_to_write)

    @torch.no_grad()
    def _get_model(self, checkpoint_path, model_params, in_channel):
        if checkpoint_path is not None:
            model_factory = ModelFactory(
                in_channels=in_channel,
                **model_params, )
            model, _ = model_factory.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                device=self.device)
        else:
            raise ValueError("No model path provided")

        return model

    def _get_model_name(self, checkpoint_path):
        return [s for s in checkpoint_path.split("/") if any(s.startswith(p) for p in ["BarlowTwin", "SimCLR"])][0]

    def _get_ID(self, checkpoint_path):
        path = [s for s in checkpoint_path.split("/") if any(s.startswith(p) for p in ["BarlowTwin", "SimCLR"])][1]
        ID = path.split("--")[-2].split(":")[1]
        return ID

    @torch.no_grad()
    def _get_projector(self, network_id, rep_dim, method=None, projector_out=None):
        if network_id in ["MnistResNet18", "MnistResNet34", "resnet10", "resnet18", "uncertainty_net"]:
            if rep_dim == 3:
                projector = (3, 3, 3)
                out_dim = 3
            elif method == "BarlowTwin" and projector_out is None:
                projector = (2048, 2048, 1024)
                out_dim = 1024
            elif method == "BarlowTwin" and projector_out is not None:
                projector = (2048, 2048, projector_out)
                out_dim = projector_out
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
        elif rep_dim <= 84:
            projector = (84, 84, 48)
        elif rep_dim <= 128:
            projector = (128, 128, 96)
        elif rep_dim <= 512:
            projector = (512, 512, 128)
            out_dim = None
        elif rep_dim <= 2048:
            projector = (2048, 2048, 128)

        else:
            projector = None
            out_dim = None

        return projector, out_dim

    @torch.no_grad()
    def vis_t_SNE(self, feats, labels, kappa, oob=None):
        # Perform PCA to reduce dimensions to a reasonable amount before performing t_SNE
        # pca = PCA(50)
        # feats = pca.fit_transform(feats)


        if oob == "cifar10":
            idx = torch.randperm(kappa.shape[0])[:4000]

        else:
            id = torch.tensor(np.random.choice(np.unique(labels.cpu().numpy()), 10), device=self.device)
            idx = torch.sum((labels[:, None] == id) * id[None], dim=-1).nonzero().squeeze()[:4000]

        feats, labels, kappa = feats[idx], labels[idx], kappa[idx]

        kappa_min = kappa.min()
        kappa_max = kappa.max()

        kappa = torch.exp((kappa - kappa_min) / (kappa_max - kappa_min) * 6)
        kappa = torch.clamp(kappa, max=500)

        if self.main:
            feats, labels, kappa = feats.cpu(), labels.cpu(), kappa.cpu()

        # Perform t-SNE
        feats_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(feats)

        if self.main:
            matplotlib.use('PDF')
        fig = plt.figure()
        plt.style.use('default')
        plt.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=labels, s=kappa, alpha=0.3)

        if oob is not None:
            data = oob
        else:
            data = self.train_data

        name = self.model_params["model_id"] + "_" + self.model_params["network_id"] + "_MC:" + \
               str(self.model_params["model_options"]["n_samples"]) + "_" + self.ID + data + "_t_SNE"
        if self.main:
            print("t-SNE about to be saved.")
            path = f"/home/lorenzni/imgs/{self.ID}"
            Path(path).mkdir(parents=True, exist_ok=True)
            fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)
            print("t-SNE should be saved now.")
        else:
            print("self.main is false. Why??")
            path = f"/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Results/imgs"
            fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)
            # plt.show()
        print(name)

    def _split_name(self, name):
        name = self._get_model_name(name)
        split = name.split("--")

        network_id = split[1]
        dataset = split[2]
        distribution = split[3]
        lambda_reg = float(split[4].split("=")[1])
        rep_dim = int(split[5])
        lambda_unc = float(split[6].split("=")[1])
        proj = split[7].split("=")[1]
        if proj == "No_projector":
            projector, out_dim = None, None
        else:
            projector_out = int(split[7].split("=")[1])
            projector, out_dim = self._get_projector(network_id, rep_dim=rep_dim, method=split[0],
                                                     projector_out=projector_out)
        ID = self._get_ID(self.checkpoint_path)

        if split[0] == "BarlowTwin":
            model_id = "barlowtwins"
            mc = str_to_bool(split[8].split("=")[1])
            lambda_bt = float(split[9].split("=")[1])
            if mc:
                n_samples = int(split[10])
            else:
                n_samples = "None"

            method_params = {"projector_hidden": projector, "lambd": lambda_bt, 'mc': mc,
                             "lambda_reg": lambda_reg,
                             "n_samples": n_samples}
        elif split[0] == "SimCLR":
            model_id = "simclr"
            loss = split[8]
            temperature = float(split[9])
            uncertainty_weighting = split[10]
            if loss == "MCInfoNCE":
                n_samples = int(split[11].split("=")[1])
                loss_method = split[12]
                reduction = split[13]

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
                        "model_options": method_params,
                        "ID": ID}

        return model_params, dataset

    def plot_kappa(self, kappa_idx, imgs, cl=0):

        u = ["low", "high"]
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        theta, phi = torch.linspace(0, 2 * torch.pi, 30), torch.linspace(0, torch.pi, 30)
        theta, phi = theta.detach().cpu(), phi.detach().cpu()
        THETA, PHI = torch.meshgrid(theta, phi, indexing="ij")
        THETA, PHI = THETA.detach().cpu(), PHI.detach().cpu()
        X, Y, Z = torch.sin(PHI) * torch.cos(THETA), torch.sin(PHI) * torch.sin(THETA), torch.cos(PHI)
        X, Y, Z = X.detach().cpu().numpy(), Y.detach().cpu().numpy(), Z.detach().cpu().numpy()

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_wireframe(X, Y, Z, linewidth=1, alpha=0.25, color="gray")

        for n, dist in enumerate(kappa_idx):
            kappa = dist.scale.detach().cpu().numpy()
            loc = dist.loc.detach().cpu().numpy()
            dist_sample = dist.rsample(n_samples=100)
            dist_sample = torch.squeeze(dist_sample)
            X, Y, Z = dist_sample[:, 0], dist_sample[:, 1], dist_sample[:, 2]
            X, Y, Z = X.detach().cpu().numpy(), Y.detach().cpu().numpy(), Z.detach().cpu().numpy()
            ax1.scatter(X, Y, Z, s=50)
            ax1.plot(*np.stack((np.zeros_like(loc), loc)).squeeze().T, linewidth=4,
                     label=f"$\kappa={kappa[0]:.{0}f}$, {u[n]} uncertainty")
            ax1.legend(loc='upper center')
            ax1.view_init(30, 45)
            ax1.tick_params(axis='both')

            if n == 0:
                ax2 = fig.add_subplot(2, 2, 2)
                ax2.imshow(imgs[n][0], interpolation='nearest')
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.tick_params(axis='both', which='both', length=0)
                ax2.set_title(f"{u[n]} kappa, class: {classes[cl]}")
            else:
                ax3 = fig.add_subplot(2, 2, 4)
                ax3.imshow(imgs[n][0], interpolation='nearest')
                plt.setp(ax3.get_xticklabels(), visible=False)
                plt.setp(ax3.get_yticklabels(), visible=False)
                ax3.tick_params(axis='both', which='both', length=0)
                ax3.set_title(f"{u[n]} kappa, class: {classes[cl]}")

        # plt.legend()
        fig.tight_layout()
        name = self.model_params["model_id"] + "_" + self.model_params["network_id"] + "_" + \
               self.model_params["model_options"]["n_samples"]
        if self.main:
            path = f"/home/lorenzni/imgs/"
        else:
            path = f"/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Reulsts/imgs/"
        fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)


if __name__ == '__main__':
    sys.path.append("")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p",
                        default="/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Cluster_runs/Barlow_Twin750/BarlowTwin--resnet18--cifar10--powerspherical--lambda_reg=0.0001--512--lambda_unc=10.0--projector_out=1048--MC=False--lambda_bt=0.0051/BarlowTwin--resnet18--cifar10--powerspherical--lambda_reg=0.0001--512--lambda_unc=10.0--projector_out=1048--MC=False--lambda_bt=0.0051--t=0.1--lambda=0.0001/epoch_750.tar",
                        type=str)
    parser.add_argument("--oob_test", "-ot", default=True, type=str2bool)
    parser.add_argument("--oob_data", "-od", default="cifar100", type=str)
    parser.add_argument('--tsne', default=True, type=str2bool)
    parser.add_argument('--main', default=False, type=str2bool)
    args = parser.parse_args()

    # path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Cluster_runs/Barlow_Twin750/BarlowTwin--resnet18--cifar10--powerspherical--lambda_reg=0.0001--512--lambda_unc=10.0--projector_out=1048--MC=False--lambda_bt=0.0051/BarlowTwin--resnet18--cifar10--powerspherical--lambda_reg=0.0001--512--lambda_unc=10.0--projector_out=1048--MC=False--lambda_bt=0.0051--t=0.1--lambda=0.0001/BarlowTwin_750.tar"
    # path = "/Users/nikolai.lorenz/Desktop/Statistik/Masterarbeit/Cluster_runs/pretrained/SimCLR--resnet18--cifar10--sphere--lambda_reg=0.0001--512--InfoNCE--0.1--/SimCLR--resnet18--cifar10--sphere/epoch_950.tar"
    V = Validate(
        checkpoint_path=args.path,
        plot_tsne=args.tsne,
        oob_test=args.oob_test,
        oob_data=args.oob_data,
        main=args.main,
    )

    Auroc, Recall, knn = V._get_metrics()
    print(f"AUROC: {Auroc:0.4f}, R@1: {Recall:0.3f}, KNN: {knn:0.1f}")
