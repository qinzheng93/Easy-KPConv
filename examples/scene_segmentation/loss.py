from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import ipdb


class LossFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output_dict, data_dict):
        scores = output_dict["scores"]
        labels = data_dict["labels"]  # list of LongTensor

        loss = self.loss(scores, labels)

        return {"loss": loss}


def evaluate_multiclass_classification(
    inputs: Tensor, targets: Tensor, dim: int, eps: float = 1e-6
) -> Tuple[Tensor, Tensor, Tensor]:
    """Multi-class classification precision and recall metric.

    This method compute overall accuracy, macro-precision and macro-recall.

    Args:
        inputs (Tensor): inputs (*, C, *)
        targets (LongTensor): targets, [0, C-1] (*)
        dim (int): the category dim.
        eps (float=1e-6): safe number

    Return:
        accuracy (Tensor): overall accuracy
        mean_precision (Tensor): mean precision over all categories.
        mean_recall (Tensor): mean recall over all categories.
    """
    num_classes = inputs.shape[dim]

    # 1. accuracy
    results = inputs.argmax(dim=dim)
    accuracy = torch.eq(results, targets).float().mean()

    # 2. precision and recall
    results = results.flatten()  # (N,)
    targets = targets.flatten()  # (N,)
    num_rows = results.shape[0]
    row_indices = torch.arange(num_rows).cuda()

    result_mat = torch.zeros(size=(num_rows, num_classes)).cuda()  # (N, C)
    result_mat[row_indices, results] = 1.0
    result_sum = result_mat.sum(dim=0)  # (C,)

    target_mat = torch.zeros(size=(num_rows, num_classes)).cuda()  # (N, C)
    target_mat[row_indices, targets] = 1.0
    target_sum = target_mat.sum(dim=0)  # (C,)

    positive_mat = result_mat * target_mat
    positive_sum = positive_mat.sum(dim=0)  # (C,)

    per_class_precision = positive_sum / (result_sum + eps)  # (C,)
    per_class_recall = positive_sum / (target_sum + eps)  # (C,)

    # 3. mask out unseen categories
    class_masks = torch.gt(target_sum, 0).float()  # (C,)
    precision = (per_class_precision * class_masks).sum() / (class_masks.sum() + eps)
    recall = (per_class_recall * class_masks).sum() / (class_masks.sum() + eps)

    return accuracy, precision, recall


def compute_metrics(output_dict, data_dict):
    # 1. accuracy, mean precision, mean recall
    labels = data_dict["labels"]
    scores = output_dict["scores"]

    accuracy, precision, recall = evaluate_multiclass_classification(scores, labels, dim=1)

    return {"accuracy": accuracy, "precision": precision, "recall": recall}
