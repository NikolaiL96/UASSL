import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy

import matplotlib
from pathlib import Path
import os

from torchmetrics.functional.classification import binary_auroc as auc
from torch.cuda.amp import autocast, GradScaler

from utils.reliability_diagrams import reliability_diagram


class Linear_Protocoler(object):

    def __init__(self, encoder, device, distribution, repre_dim: int, eval_params: dict):

        self.device = device
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)

        self.encoder = encoder  # .to(self.device)
        # Set to evaluation mode
        self.encoder.eval()
        self.repre_dim = repre_dim
        self.train_features = None
        self.train_labels = None
        self.eval_params = eval_params
        self.distribution = distribution

    @torch.no_grad()
    def recall_auroc(self, test_dl, get_entropy=False):
        Recall = []
        Auroc = []
        Uncertainty, Loc, Labels = (), (), ()
        for x, labels in test_dl:
            x, labels = x.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                feats = self.encoder(x)

            labels = labels

            if self.distribution not in ["normal", "normalSingleScale"]:
                unc = 1 / feats.scale
            else:
                unc = feats.scale

            loc = feats.loc

            loc = F.normalize(loc, dim=-1)
            closest_idxes = loc.matmul(loc.transpose(-2, -1)).topk(2)[1][:, 1]
            closest_classes = labels[closest_idxes]
            is_same_class = (closest_classes == labels).float()
            auroc = auc(-unc.squeeze(), is_same_class.int()).item()

            Uncertainty += (unc,)
            Loc += (loc,)
            Labels += (labels,)

            Recall.append(is_same_class.mean())
            Auroc.append(auroc)

        Uncertainty = torch.cat(Uncertainty)
        Loc = torch.cat(Loc)
        Labels = torch.cat(Labels)


        Recall = torch.stack(Recall, 0)
        Auroc = torch.Tensor(Auroc)

        c_idx = Loc.matmul(Loc.transpose(-2, -1)).topk(2)[1][:, 1]
        c_c = Labels[c_idx]
        correctness = (c_c == Labels).float()
        Auroc_set = auc(-Uncertainty.squeeze(), correctness.int()).item()

        if get_entropy:
            entropy_uncertainty = entropy(Uncertainty.cpu().numpy(), axis=0)
            print(f"Entropy uncertainty: {entropy_uncertainty}")

        return Recall.mean(), Auroc.mean(), correctness.mean(), Auroc_set

    @torch.no_grad()
    def knn_accuracy(self, train_dl, test_dl, knn_k: int = 200, knn_t: float = 0.1):
        train_features, train_labels = self._knn_extract_features_and_labels(
            train_dl=train_dl)

        accuracy = self._knn_predict_with_given_features_and_labels(
            train_features=train_features,
            train_labels=train_labels,
            test_dl=test_dl,
            knn_k=knn_k,
            knn_t=knn_t
        )

        return accuracy

    @torch.no_grad()
    def _knn_extract_features_and_labels(self, train_dl):
        # extract train
        train_features, train_labels = (), ()

        for x, labels in train_dl:
            x, labels = x.to(self.device), labels.to(self.device)
            with autocast(enabled=self.use_amp):
                feats = self.encoder(x)

            feats = feats.loc
            train_features += (F.normalize(feats, dim=1),)
            train_labels += (labels,)

        train_features = torch.cat(train_features).t().contiguous()
        train_labels = torch.cat(train_labels)

        return train_features, train_labels

    @torch.no_grad()
    def _knn_predict_with_given_features_and_labels(self, train_features, train_labels, test_dl, knn_k: int = 200,
                                                    knn_t: float = 0.1, num_classes=None):

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

    def train(self, dataloader):
        optim_params = self.eval_params["optim_params"]
        num_epochs = self.eval_params["num_epochs"]
        schedular_params = {"T_max": 100 * len(dataloader), "eta_min": 0.001}

        # get classes
        num_classes = len(dataloader.dataset.classes)

        # Define classifier
        self.classifier = nn.Linear(self.repre_dim, num_classes).to(self.device)

        # Define optimizer
        optimizer = SGD(self.classifier.parameters(), **optim_params)
        # Define loss
        ce_loss = nn.CrossEntropyLoss()

        # Define scheduler
        scheduler = CosineAnnealingLR(optimizer, **schedular_params)

        # Train
        for epoch in range(num_epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # forward
                with autocast(enabled=self.use_amp):
                    dist = self.encoder(x)

                x = dist.loc
                with autocast(enabled=self.use_amp):
                    loss = ce_loss(self.classifier(x), y)

                # backward
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            if scheduler:
                scheduler.step()

    def linear_accuracy(self, dataloader, epoch=None):
        """
        Returns the accuracy of the Linear Protocoler on the given dataloader

        returns:
            accuracy: float Correct predictions / Total predictions

        """
        if epoch is not None:
            true_labels, pred_labels, confidences = [], [], []

        total_top1, total_num = 0.0, 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            self.classifier.eval()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # calculate outputs by running images through the network
                with autocast(enabled=self.use_amp):
                    dist = self.encoder(x)

                loc = dist.loc

                with autocast(enabled=self.use_amp):
                    outputs = self.classifier(loc)

                # the class with the highest energy is what we choose as prediction1
                _, predicted = torch.max(outputs.data, 1)

                total_num += y.size(0)
                total_top1 += (predicted == y).float().sum().item()

                if epoch is not None:
                    true_labels.extend(y.cpu().numpy())
                    pred_labels.extend(predicted.cpu().numpy())
                    confidences.extend(dist.scale.cpu().numpy())

        self.classifier.train()

        if epoch is not None:
            true_labels = np.asarray(true_labels, dtype=np.int64)
            pred_labels = np.asarray(pred_labels, dtype=np.int64)
            confidences = np.asarray(confidences, dtype=np.float32)

            if self.distribution in ["normal", "normalSingleScale"]:
                # Confidence is certainty not uncertainty
                confidences = 1 / confidences

            confidences_minmax = (confidences - confidences.min()) / (confidences.max() - confidences.min())

            matplotlib.use('PDF')
            fig, ece = reliability_diagram(true_labels, pred_labels, confidences_minmax, num_bins=20, return_fig=True)

            id = os.getenv('SLURM_JOB_ID')
            name = f"ReliabilityPlot_Epoch_{epoch}"
            path = f"/home/lorenzni/imgs/{id}"
            Path(path).mkdir(parents=True, exist_ok=True)
            fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)
            return total_top1 / total_num * 100, ece

        return total_top1 / total_num * 100

    def linear_uncertainty(self, dataloader, epoch="", M=20):
        total = 0
        nll = 0
        sample_mean_acc = 0

        true_labels, pred_labels, confidences = [], [], []

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            self.classifier.eval()

            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # calculate outputs by running images through the network
                with autocast(enabled=self.use_amp):
                    dist = self.encoder(x)

                # Take M=20 monte Carlo samples
                shape = dist.loc.shape
                samples = dist.rsample(torch.Size([M])).reshape(
                    M * shape[0], shape[1])

                with autocast(enabled=self.use_amp):
                    outputs = self.classifier(samples).reshape(M, shape[0], -1)

                # We take the mean value of the logits
                out_logits = torch.mean(outputs, dim=0)
                out_prob = torch.softmax(out_logits, dim=1)

                logprob_mean = torch.log(torch.gather(out_prob, 1, y.unsqueeze(1)) + 1e-6)
                nll -= torch.sum(logprob_mean).item()
                total += y.size(0)

                prob, pred_class = torch.max(out_prob, dim=1)
                sample_mean_acc += (pred_class == y).float().sum().item()

                true_labels.extend(y.cpu().numpy())
                pred_labels.extend(pred_class.cpu().numpy())
                confidences.extend(prob.cpu().numpy())

            self.classifier.train()
            self.encoder.train()

        true_labels = np.asarray(true_labels, dtype=np.int64)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        confidences = np.asarray(confidences, dtype=np.float32)
        matplotlib.use('PDF')
        fig, ece = reliability_diagram(
            true_labels, pred_labels, confidences, num_bins=20, return_fig=True
        )

        id = os.getenv('SLURM_JOB_ID')
        name = f"ReliabilityPlot_Epoch_{epoch}"
        path = f"/home/lorenzni/imgs/{id}"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/{name}.pdf', dpi=fig.dpi)

        # We return: the nll of the sampling mean probability and the accuracy after aggregating the samples
        return nll / total, sample_mean_acc / total, ece


# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(
        feature: torch.Tensor,
        feature_bank: torch.Tensor,
        feature_labels: torch.Tensor,
        num_classes: int,
        knn_k: int,
        knn_t: float,
) -> torch.Tensor:
    """Run kNN predictions on features based on a feature bank
    This method is commonly used to monitor performance of self-supervised
    learning methods.
    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.
    Args:
        feature:
            Tensor of shape [N, D] for which you want predictions
        feature_bank:
            Tensor of a database of features used for kNN
        feature_labels:
            Labels for the features in our feature_bank
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k:
            Number of k neighbors used for kNN
        knn_t:
            Temperature parameter to reweights similarities for kNN
    Returns:
        A tensor containing the kNN predictions
    Examples:
        images, targets, _ = batch
        feature = backbone(images).squeeze()
        # we recommend to normalize the features
        feature = F.normalize(feature, dim=1)
        pred_labels = knn_predict(
             feature,
             feature_bank,
             targets_bank,
             num_classes=10,
         )
    """

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
