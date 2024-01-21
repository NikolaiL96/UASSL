"""Implementation of AUROC metric based on TorchMetrics."""
import torch
from torch import Tensor
from torchmetrics import ROC
# from torchmetrics.functional import auc
#from torchmetrics.functional.classification import binary_auroc as auc
#import torch.nn.functional as F


class AUROC(ROC):
    """Area under the ROC curve."""

    def compute(self) -> Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            Value of the AUROC metric
        """
        tpr: Tensor
        fpr: Tensor

        fpr, tpr, _thresholds = super().compute()
        raise NotImplementedError("Needs to be updated")
        # TODO: use stable sort after upgrading to pytorch 1.9.x (https://github.com/openvinotoolkit/anomalib/issues/92)
        if not (torch.all(fpr.diff() <= 0) or torch.all(fpr.diff() >= 0)):
            return auc(fpr, tpr, reorder=True)  # only reorder if fpr is not increasing or decreasing
        return auc(fpr, tpr)


def recall_at_one(features, targets, mode="matmul"):
    if mode == "matmul":
        # Expects tensors as inputs
        features = F.normalize(features, dim=-1)
        closest_idxes = features.matmul(features.transpose(-2, -1)).topk(2)[1][:, 1]
        closest_classes = targets[closest_idxes]
        is_same_class = (closest_classes == targets).float()
    return is_same_class.mean(), is_same_class


def auroc(uncertainties, correctness):
    auroc_correct = auc(-uncertainties, torch.from_numpy(correctness).int()).item()
    return auroc_correct
