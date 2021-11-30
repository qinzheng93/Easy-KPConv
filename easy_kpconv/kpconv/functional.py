import torch

from ..utils import index_select


def nearest_upsample(x, upsample_indices):
    """Pools features from the closest neighbors.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        x: [n1, d] features matrix
        upsample_indices: [n2, max_num] Only the first column is used for pooling

    Returns:
        x: [n2, d] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), dim=0)
    x = index_select(x, upsample_indices[:, 0], dim=0)
    return x


def maxpool(x, neighbor_indices):
    """Max pooling from neighbors.

    Args:
        x: [n1, d] features matrix
        neighbor_indices: [n2, max_num] pooling indices

    Returns:
        pooled_feats: [n2, d] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), dim=0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    pooled_feats = neighbor_feats.max(1)[0]
    return pooled_feats


def global_avgpool(x, lengths):
    """Global average pooling over batch.

    Args:
        x: [N, D] input features
        batch_lengths: [B] list of batch lengths

    Returns:
        x: [B, D] averaged features
    """
    averaged_features = []
    start_index = 0
    for _, length in enumerate(lengths):
        end_index = start_index + length
        averaged_features.append(torch.mean(x[start_index:end_index], dim=0))
        start_index = end_index
    x = torch.stack(averaged_features, dim=0)
    return x
