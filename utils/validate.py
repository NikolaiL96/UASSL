import numpy as np
from scipy.stats import entropy, spearmanr
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc as auc
import torchvision.transforms.functional as TF

from sklearn.manifold import TSNE

import os

from utils import load_dataset
from utils import Linear_Protocoler

from utils.utils import _find_low_and_high_images, get_cifar10h, knn_predict


class Validate:

    def __init__(self, data, distribution, model, epoch=None, last_epoch=False,
                 oob_data=None, oob_test=False, plot_tsne=False):


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.last_epoch = last_epoch
        self.oob_test = oob_test
        self.plot_tsne = plot_tsne

        self.data = data
        self.dl_kwargs = {"batch_size": 512, "shuffle": True, "num_workers": min(os.cpu_count(), 0)}

        if oob_data is not None:
            self.data_test, *_ = load_dataset(oob_data, "./data/",
                                              augmentation_type="BYOL",
                                              dl_kwargs=self.dl_kwargs)
        else:
            self.data_test = self.data

        self.model = model
        self.encoder = self.model.backbone_net
        self.encoder.eval()

        self.epoch = epoch
        self.distribution = distribution

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
        linear_evaluator = Linear_Protocoler(self.model.backbone_net.eval(),
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

                if self.distribution in ["sphere", "normal"]:
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

                if self.distribution not in ["sphere", "normal"]:
                    kappa = 1 / kappa
                else:
                    kappa = kappa

                recall, auroc = self._get_roc(loc, labels, kappa)

                test_labels += (labels.cpu(),)
                test_kappa += (kappa,)
                test_loc += (loc,)
                Recall.append(recall)
                Auroc.append(auroc)


            Labels = torch.cat(test_labels).to(self.device)
            U = torch.cat(test_kappa).to(self.device)
            Loc = torch.cat(test_loc).to(self.device)
            self.vis_t_SNE(Loc, Labels, U, dataset)

            Recall = torch.stack(Recall, 0).mean()
            Auroc = torch.Tensor(Auroc).mean()

        return Recall, Auroc


    @torch.no_grad()
    def vis_t_SNE(self, feats, labels, kappa, data=None):

        if data == "cifar10":
            idx = torch.randperm(kappa.shape[0])[:4000]

        else:
            id = torch.tensor(np.random.choice(np.unique(labels.cpu().numpy()), 10), device=self.device)
            idx = torch.sum((labels[:, None] == id) * id[None], dim=-1).nonzero().squeeze()[:4000]

        feats, labels, kappa = feats[idx], labels[idx], kappa[idx]

        kappa_min = kappa.min()
        kappa_max = kappa.max()

        kappa = torch.exp((kappa - kappa_min) / (kappa_max - kappa_min) * 6)
        kappa = torch.clamp(kappa, max=500)

        feats, labels, kappa = feats.cpu(), labels.cpu(), kappa.cpu()

        # Perform t-SNE
        feats_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(feats)

        matplotlib.use('PDF')
        fig = plt.figure()
        plt.style.use('default')
        plt.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=labels, s=kappa, alpha=0.3)

        id = os.getenv('SLURM_JOB_ID')
        name = f"t-SNE_{data}_Epoch_{self.epoch}"
        path = f"/home/lorenzni/imgs/{id}"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)

