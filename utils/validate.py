import numpy as np
from scipy.stats import entropy, spearmanr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc as auc
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast

from sklearn.manifold import TSNE

import os

from utils import load_dataset
from utils import Linear_Protocoler

from utils.utils import get_cifar10h, knn_predict


class Validate:

    def __init__(self, data, distribution, model, device, epoch=None, last_epoch=False, low_shot=False,
                 plot_tsne=False):

        self.device = device
        self.use_amp = device.type == 'cuda'

        self.last_epoch = last_epoch
        self.low_shot = low_shot
        self.plot_tsne = plot_tsne
        self.data = data

        dataset = str(data.test_dl.dataset).split("\n")[0].split(" ")[1]
        oob_data = "cifar10" if dataset.lower() == "cifar100" else "cifar100"

        if low_shot:
            dl_kwargs = {"batch_size": 512, "shuffle": True, "num_workers": min(os.cpu_count(), 0)}
            self.data_test, *_ = load_dataset(oob_data, "./data/", augmentation_type="BYOL", dl_kwargs=dl_kwargs)
        else:
            self.data_test = self.data

        self.model = model
        self.encoder = self.model.backbone_net
        self.encoder.eval()
        self.model.eval()

        self.epoch = epoch
        self.distribution = distribution

    @torch.no_grad()
    def _get_roc(self, feats, labels, unc):
        feats = F.normalize(feats, dim=-1)

        # Calculated class of nearest neighbor
        closest_idx = feats.matmul(feats.transpose(-2, -1)).topk(2)[1][:, 1]
        closest_classes = labels[closest_idx]

        is_same_class = closest_classes == labels
        auroc = auc(-unc.squeeze(), is_same_class.int())

        return torch.as_tensor(is_same_class, dtype=torch.float, device=self.device).mean(), torch.as_tensor(auroc)

    @torch.no_grad()
    def get_cor(self):
        test_unc = ()
        dl_kwargs = {"batch_size": 512, "shuffle": False, "num_workers": min(os.cpu_count(), 0)}
        data_cifar10h, *_ = load_dataset("cifar10", "./data/", augmentation_type="BYOL", dl_kwargs=dl_kwargs)

        for n, (x_test, labels_test) in enumerate(data_cifar10h.test_dl):
            x_test, labels_test = x_test.to(self.device), labels_test.to(self.device)

            with autocast(enabled=self.use_amp):
                feats_test = self.encoder(x_test)

            unc_test = feats_test.scale
            test_unc += (unc_test,)

        unc = torch.cat(test_unc).cpu().numpy()
        if self.distribution not in ["normal", "normalSingleScale"]:
            unc = 1 / unc

        try:
            cifar10h = get_cifar10h()
            unc_h = entropy(cifar10h, axis=1)

            corr = np.corrcoef(unc, unc_h)
            rank_corr = spearmanr(unc, unc_h)[0]

            return corr[0, 1], rank_corr

        except Exception as e:
            print(e)
            return 0., 0.

    def get_linear_probing(self, eval_params, epoch=None):
        if not self.low_shot:
            test_loader = self.data.test_dl
        else:
            test_loader = self.data_test.test_dl
        linear_evaluator = Linear_Protocoler(self.model.backbone_net.eval(), repre_dim=self.model.rep_dim,
                                             device=self.device, eval_params=eval_params, distribution=self.distribution)

        linear_evaluator.train(self.data.train_eval_dl)
        return linear_evaluator.linear_accuracy(test_loader, epoch)

    @torch.no_grad()
    def _extract_train(self, train_dl):
        # extract train
        train_features, train_labels, train_dist = (), (), ()

        for x, labels in train_dl:
            x, labels = x.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                feats = self.encoder(x)

            train_features += (F.normalize(feats.loc, dim=1),)
            train_labels += (labels,)

        train_features = torch.cat(train_features).t().contiguous()
        train_labels = torch.cat(train_labels)

        return train_features, train_labels

    @torch.no_grad()
    def recall_auroc(self, test_dl=None):
        dl = self.data_test.test_dl if test_dl is None else test_dl

        Recall = []
        Auroc = []
        for x, labels in dl:
            x, labels = x.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                feats = self.encoder(x)

            labels = labels
            unc = 1 / feats.scale
            feats = feats.loc

            feats = F.normalize(feats, dim=-1)
            closest_idxes = feats.matmul(feats.transpose(-2, -1)).topk(2)[1][:, 1]
            closest_classes = labels[closest_idxes]
            is_same_class = (closest_classes == labels).float()
            auroc = auc(-unc.squeeze(), is_same_class.int()).item()

            Recall.append(is_same_class.mean())
            Auroc.append(auroc)

        Recall = torch.stack(Recall, 0)
        Auroc = torch.Tensor(Auroc)
        return Recall.mean(), Auroc.mean()

    @torch.no_grad()
    def knn_accuracy(self, knn_k: int = 200, knn_t: float = 0.1):
        train_features, train_labels = self._extract_train(train_dl=self.data.train_eval_dl)

        accuracy = self._knn_predict_with_given_features_and_labels(
            train_features=train_features,
            train_labels=train_labels,
            test_dl=self.data_test.test_dl,
            knn_k=knn_k,
            knn_t=knn_t)

        return accuracy

    @torch.no_grad()
    def get_metrics(self):

        test_labels, test_uncertainty, test_loc = (), (), ()

        for n, (x, labels) in enumerate(self.data_test.test_dl):
            x, labels = x.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                feats = self.encoder(x)

            loc = feats.loc
            if self.distribution not in ["normal", "normalSingleScale"]:
                uncertainty = 1 / feats.scale
            else:
                uncertainty = feats.scale

            if self.plot_tsne:
                test_labels += (labels,)
                test_uncertainty += (uncertainty,)
                test_loc += (loc,)

        if self.last_epoch:
            p_corrupted, cor_corrupted = self.corrupted_img()
        else:
            p_corrupted, cor_corrupted = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        if self.plot_tsne:
            dataset = str(self.data_test.test_dl.dataset).split("\n")[0].split(" ")[1]
            self.vis_t_SNE(torch.cat(test_loc), torch.cat(test_labels), torch.cat(test_uncertainty), dataset)

        return cor_corrupted, p_corrupted

    @torch.no_grad()
    def corrupted_img(self):
        # Modified from https://github.com/mkirchhof/url/tree/main
        crop_min = torch.tensor(0.1, device=self.device)
        crop_max = torch.tensor(0.5, device=self.device)
        crop = torch.rand(len(self.data_test.test_dl.dataset), device=self.device) * (crop_max - crop_min) + crop_min

        unc, unc_c = (), ()

        for n, (x, target) in enumerate(self.data_test.test_dl):
            x, target = x.to(self.device), target.to(self.device)
            x_c = torch.zeros_like(x, device=self.device)
            c_sizes = torch.zeros(x.shape[0], device=self.device)

            for i in range(x.shape[0]):
                # Crop each image individually because torchvision cannot do it batch-wise
                crop_size = int(torch.round(min(x.shape[2], x.shape[3]) * crop[n * x.shape[0] + i]))
                c_sizes[i] = crop_size
                c = TF.center_crop(x[i], [crop_size])
                x_c[i] = TF.resize(c, [x.shape[2], x.shape[3]], antialias=True)

            with autocast(enabled=self.use_amp):
                unc += (self.encoder(x).scale,)
                unc_c += (self.encoder(x_c).scale,)

        unc, unc_c = torch.cat(unc), torch.cat(unc_c)

        if self.distribution not in ["normal", "normalSingleScale"]:
            unc, unc_c = 1 / unc, 1 / unc_c


        p_cropped = (unc > unc_c).float().mean()
        cor_cropped = spearmanr(-unc_c.cpu().numpy(), crop.cpu().numpy())[0]

        return p_cropped, cor_cropped

    @torch.no_grad()
    def get_zero_shot_metrics(self, dataset):
        dl_kwargs = {"batch_size": 512, "shuffle": True, "num_workers": min(os.cpu_count(), 0)}
        if dataset == "sop":
            ssl_data, _, _ = load_dataset("sop", "/home/lorenzni/data/online_products/images/", "BYOL",
                                          dl_kwargs=dl_kwargs)
        elif dataset == "cup200":
            ssl_data, _, _ = load_dataset("cup200", "/home/lorenzni/data/cub200/images", "BYOL",
                                          dl_kwargs=dl_kwargs)
        elif dataset == "cars196":
            ssl_data, _, _ = load_dataset("cars196", "/home/lorenzni/data/cars196/images", "BYOL",
                                          dl_kwargs=dl_kwargs)

        Recall = []
        Auroc = []

        test_labels = ()
        test_kappa = ()
        test_loc = ()

        for n, (x, labels) in enumerate(ssl_data.test_dl):
            x, labels = x.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                feats = self.encoder(x)

            labels = labels
            kappa = feats.scale
            loc = feats.loc

            if self.distribution not in ["normal", "normalSingleScale"]:
                kappa = 1 / kappa
            else:
                kappa = kappa

            recall, auroc = self._get_roc(loc, labels, kappa)

            test_labels += (labels.cpu(),)
            test_kappa += (kappa,)
            test_loc += (loc,)
            Recall.append(recall)
            Auroc.append(auroc)

            Labels = torch.cat(test_labels)
            U = torch.cat(test_kappa)
            Loc = torch.cat(test_loc)
            self.vis_t_SNE(Loc, Labels, U, dataset)

            Recall = torch.stack(Recall, 0).mean()
            Auroc = torch.Tensor(Auroc).mean()

        return Recall, Auroc

    @torch.no_grad()
    def vis_t_SNE(self, feats, labels, unc, data=None):

        # if data == "cifar10":
        #     idx = torch.randperm(unc.shape[0])[:10000]
        #
        # else:
        #     id = torch.tensor(np.random.choice(np.unique(labels.cpu().numpy()), 10), device=self.device)
        #     idx = torch.sum((labels[:, None] == id) * id[None], dim=-1).nonzero().squeeze()[:4000]

        #id = torch.tensor(np.random.choice(np.unique(labels.cpu().numpy()), 2), device=self.device)

        id = torch.tensor([1, 9], device=self.device)
        idx = torch.sum((labels[:, None] == id) * id[None], dim=-1).nonzero().squeeze()[:4000]

        feats, labels, unc = feats[idx], labels[idx], unc[idx]

        unc_min = unc.min()
        unc_max = unc.max()
        # Exponential weighting of higher uncertainties for better visualization
        unc = torch.exp((unc - unc_min) / (unc_max - unc_min) * 3.5)


        feats, labels, unc = feats.cpu(), labels.cpu(), unc.cpu()

        # Perform t-SNE
        feats_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(feats)

        colors = ['#b5de73', '#99a9cf']
        cmap = ListedColormap(colors)

        matplotlib.use('PDF')
        fig = plt.figure(figsize=(6, 10))
        plt.style.use('default')
        plt.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=labels, cmap=cmap, s=10, alpha=0.6)

        # Remove axis ticks and labels.
        plt.xticks([])
        plt.yticks([])

        id = os.getenv('SLURM_JOB_ID')
        name = f"t-SNE_{data}_Epoch_{self.epoch}"
        path = f"/home/lorenzni/imgs/{id}"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)

    @torch.no_grad()
    def _knn_predict_with_given_features_and_labels(
            self,
            train_features,
            train_labels,
            test_dl,
            knn_k: int = 200,
            knn_t: float = 0.1,
            num_classes=None
    ):
        if num_classes is None:
            num_classes = len(set(train_labels.detach().cpu().numpy().tolist()))

        # Test
        total_top1, total_num = 0.0, 0
        for x, target in test_dl:
            x, target = x.to(self.device), target.to(self.device)

            with autocast(enabled=self.use_amp):
                features = self.encoder(x)

            features = features.loc
            features = F.normalize(features, dim=1)

            # Get knn predictions
            pred_labels = knn_predict(features, train_features, train_labels, num_classes, knn_k, knn_t)

            total_num += x.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

        return total_top1 / total_num * 100
