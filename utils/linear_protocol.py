from copy import deepcopy
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc as auc
from torch.cuda.amp import autocast

class Linear_Protocoler(object):
    def __init__(
            self, encoder, repre_dim: int, variational: bool = True, device: str = "cuda", exclusion_mode='none'
    ):
        self.device = device
        self.variational = variational
        self.use_amp = device.type == 'cuda'

        self.encoder = encoder  # .to(self.device)
        # Set to evaluation mode
        self.encoder.eval()
        self.repre_dim = repre_dim
        self.train_features = None
        self.train_labels = None
        self.exclusion_mode = exclusion_mode

    @torch.no_grad()
    def recall_auroc(self, test_dl):
        Recall = []
        Auroc = []
        for x, labels in test_dl:
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    feats = self.encoder(x.to(self.device))
                labels = labels.to(self.device)
                dist = 1 / feats.scale
                feats = feats.loc

                feats = F.normalize(feats, dim=-1)
                closest_idxes = feats.matmul(feats.transpose(-2, -1)).topk(2)[1][:, 1]
                closest_classes = labels[closest_idxes]
                is_same_class = (closest_classes == labels).float()
                auroc = auc(-dist.squeeze(), is_same_class.int()).item()

                Recall.append(is_same_class.mean())
                Auroc.append(auroc)

        Recall = torch.stack(Recall, 0)
        Auroc = torch.Tensor(Auroc)
        return Recall.mean(), Auroc.mean()


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
    def auroc(self, uncertainties, correctness):
        auroc_correct = auc(-uncertainties, correctness.int()).item()
        return auroc_correct

    @torch.no_grad()
    def _get_uncertainties(self, dl):
        for n, (x, y) in enumerate(dl):
            x = x.to(self.device)
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    x = self.encoder(x)
                x = x.scale
            if n != 0:
                dist = torch.cat([dist, x])
            else:
                dist = x
        uncertainties = dist.cpu().numpy()
        return uncertainties

    @torch.no_grad()
    def _feats_labels_uncertanties(self, dl):
        Features = ()
        Labels = ()
        Dist = ()
        for x, labels in dl:
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    feats = self.encoder(x.to(self.device))
                dist = feats.scale
                feats = feats.rsample()

            Features += (feats,)
            Labels += (labels,)
            Dist += (dist,)

        Features = torch.cat(Features).t().contiguous()
        Labels = torch.cat(Labels).to(self.device)
        Dist = torch.cat(Dist).to(self.device)
        return Features, Labels, Dist


    @torch.no_grad()
    def _knn_extract_features_and_labels(self, train_dl, uncertaities=False):
        # extract train
        train_features = ()
        train_labels = ()
        train_dist = ()
        for x, labels in train_dl:
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    feats = self.encoder(x.to(self.device))
                dist = feats.scale
            if self.variational:
                feats = feats.mean
            train_features += (F.normalize(feats, dim=1),)
            train_labels += (labels,)
            train_dist += (dist,)
        #self.train_features = torch.cat(train_features).t().contiguous()
        #self.train_labels = torch.cat(train_labels).to(self.device)
        #return self.train_features, self.train_labels
        train_features = torch.cat(train_features).t().contiguous()
        train_labels = torch.cat(train_labels).to(self.device)
        train_dist = torch.cat(train_dist).to(self.device)
        if uncertaities:
            return train_features, train_labels, train_dist
        else:
            return train_features, train_labels

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
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    features = self.encoder(x)

            if self.variational:
                features = features.mean
            features = F.normalize(features, dim=1)

            # Get knn predictions
            pred_labels = knn_predict(
                features, train_features, train_labels, num_classes, knn_k, knn_t
            )

            total_num += x.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

        return total_top1 / total_num * 100

    def train(self, dataloader, num_epochs, lr, milestones=None):
        # get classes
        num_classes = len(dataloader.dataset.classes)

        # Define classifier
        self.classifier = nn.Linear(
            self.repre_dim, num_classes).to(self.device)

        # Define optimizer
        optimizer = opt.Adam(self.classifier.parameters(), lr)
        # Define loss
        ce_loss = nn.CrossEntropyLoss()
        # Define scheduler
        if milestones:
            scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones)
        else:
            scheduler = None


        #Maybe we want to use the preloaded features:
        #if self.train_features is None:
        #    self._knn_extract_features_and_labels(dataloader)
        #print("Evaluation has the following in shapes:", self.train_labels.shape,self.train_features.shape)
        batch_size = dataloader.batch_size
        # Train
        for epoch in range(num_epochs):

            #indices = torch.randperm(self.train_features.shape[1])
            #for i in range(0,len(indices),batch_size):
            #    idxs = indices[i*batch_size:(i+1)*batch_size]
            #    x = self.train_features[:, idxs].to(self.device)
            #    y = self.train_labels[idxs].to(self.device)
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # forward
                with torch.no_grad():
                    with autocast(enabled=self.use_amp):
                        dist = self.encoder(x)
                if self.variational:
                    # x = x.mean
                    x = dist.rsample()
                else:
                    x = dist

                if self.exclusion_mode != 'none':
                    mask = self.filter_uncertainty(dist)
                    x = x * mask[:, None]
                    y = y * mask[:]
                loss = ce_loss(self.classifier(x), y)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

    def filter_uncertainty(self, dist, inclusion_p=0.8):
        s1 = dist.scale

        if self.exclusion_mode == 'random':
            mask = torch.rand((len(s1))) < inclusion_p
        elif self.exclusion_mode == 'concentration':
            # remove part of the batch
            threshold = torch.topk(s1, k=int((1 - inclusion_p) * len(s1)),
                                   largest=False, sorted=True).values[-1]
            mask = (s1 > threshold)
        elif self.exclusion_mode == 'concentration-sampling':
            kappa = s1
            inclusion_scale = 1
            mu = torch.mean(kappa)
            sigma = torch.std(kappa)
            norm_kappa = (kappa - mu) / sigma * inclusion_scale

            bias = (1 + np.pi * inclusion_scale ** 2 / 8) ** 0.5 * np.log(inclusion_p / (1 - inclusion_p))
            p = torch.sigmoid(norm_kappa + bias)
            mask = torch.bernoulli(p).bool()
        else:
            raise ValueError('Unknown exclusion mode', self.exclusion_mode)
        return mask

    def linear_accuracy(self, dataloader):
        """
        Returns the accuracy of the Linear Protocoler on the given dataloader

        returns:
            accuracy: float Correct predictions / Total predictions

        """
        total_top1, total_num = 0.0, 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            self.classifier.eval()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # calculate outputs by running images through the network
                with torch.no_grad():
                    with autocast(enabled=self.use_amp):
                        x = self.encoder(x)
                if self.variational:
                    x = x.mean

                outputs = self.classifier(x)
                # the class with the highest energy is what we choose as prediction1
                _, predicted = torch.max(outputs.data, 1)

                total_num += y.size(0)
                total_top1 += (predicted == y).float().sum().item()
        self.classifier.train()

        return total_top1 / total_num * 100

    def linear_uncertainty(self, dataloader, reliability_name="", M=20):
        total = 0
        nll = 0
        sample_mean_acc = 0
        ece = -1
        if reliability_name:
            true_labels, pred_labels, confidences = [], [], []

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            self.classifier.eval()
            # TODO maybe combine it with the linear_accuracy to save some time
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # calculate outputs by running images through the network
                with torch.no_grad():
                    with autocast(enabled=self.use_amp):
                        x = self.encoder(x)
                assert (
                    self.variational
                ), "Linear uncertainty is only defined for variational models"
                dis = x

                # Take M=20 monte Carlo samples
                shape = dis.mean.shape
                samples = dis.rsample(torch.Size([M])).reshape(
                    M * shape[0], shape[1])
                outputs = self.classifier(samples).reshape(M, shape[0], -1)

                # We take the mean value of the logits
                out_logits = torch.mean(outputs, dim=0)
                out_prob = torch.softmax(out_logits, dim=1)

                # Aggregating before softmax had better results:
                # We can use a softmax per sample to get probabilities
                # out_prob = torch.softmax(outputs, dim=2)
                # Calculate NLL on the average probability over the Samples
                # out_prob_mean = torch.mean(out_prob, dim=0)

                logprob_mean = torch.log(
                    torch.gather(out_prob, 1, y.unsqueeze(1)) + 1e-6
                )
                nll -= torch.sum(logprob_mean).item()
                total += y.size(0)

                prob, pred_class = torch.max(out_prob, dim=1)
                sample_mean_acc += (pred_class == y).float().sum().item()

                if reliability_name:
                    true_labels.extend(y.cpu().numpy())
                    pred_labels.extend(pred_class.cpu().numpy())
                    confidences.extend(prob.cpu().numpy())

            self.classifier.train()
            self.encoder.train()

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
