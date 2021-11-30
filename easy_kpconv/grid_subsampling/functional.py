import torch
import numpy as np
from pykeops.torch import Vi, Vj


def keops_knn(k, dimension=3):
    r"""K nearest neighbors using PyKeOps."""
    xi = Vi(0, dimension)
    xj = Vj(1, dimension)
    dij = (xi - xj).sqnorm2()
    knn_func = dij.Kmin_argKmin(k, dim=1)
    return knn_func


@torch.no_grad()
def pad_batch_indices(points, lengths, inf=1e12):
    r"""Pad points with batch indices to enable batch radius search."""
    batch_size = len(lengths)
    batch_indices = []
    for i in range(batch_size):
        cur_batch_indices = torch.full((lengths[i], 1), inf * i).cuda()
        batch_indices.append(cur_batch_indices)
    batch_indices = torch.cat(batch_indices, dim=0)
    points = torch.cat([batch_indices, points], dim=1)
    return points


@torch.no_grad()
def batch_radius_search(q_points, s_points, q_lengths, s_lengths, radius, max_neighbor, inf=1e12):
    r"""Batch radius search.

    For each point in `q_points`, find its neighbors in `s_points` which are closer than `radius` to it. Only the
    closest `max_neighbor` neighbors are reserved.

    Args:
        q_points: torch.Tensor (N, 3), query points.
        s_points: torch.Tensor (M, 3), support points.
        q_lengths: torch.Tensor (B,), the numbers of query points in the batch.
        s_lengths: torch.Tensor (B,), the numbers of support points in the batch.
        radius: float, search neighbors within this radius for each query point.
        max_neighbor: int, maximum number of neighbors.
        inf (optional): float, batch stride, default: 1e12.
    
    Returns:
        knn_indices: torch.LongTensor (N, max_neighbor), the indices in `s_points` of the neighbors of each query point.
            If there are less than `max_neighbor` neighbors, the indices are filled with M.
    """
    knn_func = keops_knn(max_neighbor, dimension=4)
    q_points = pad_batch_indices(q_points, q_lengths, inf=inf)
    s_points = pad_batch_indices(s_points, s_lengths, inf=inf)
    knn_distances, knn_indices = knn_func(q_points, s_points)
    knn_masks = torch.lt(knn_distances, radius ** 2)
    knn_indices.masked_fill_(~knn_masks, s_points.shape[0])
    return knn_indices


@torch.no_grad()
def radius_search(q_points, s_points, q_lengths, s_lengths, radius, max_neighbor):
    r"""Batch radius search.

    Slow version of `batch_radius_search`.

    For each point in `q_points`, find its neighbors in `s_points` which are closer than `radius` to it. Only the
    closest `max_neighbor` neighbors are reserved.

    Args:
        q_points: torch.Tensor (N, 3), query points.
        s_points: torch.Tensor (M, 3), support points.
        q_lengths: list of ints (B,), the numbers of query points in the batch.
        s_lengths: list of ints (B,), the numbers of support points in the batch.
        radius: float, search neighbors within this radius for each query point.
        max_neighbor: int, maximum number of neighbors.
        inf (optional): float, batch stride, default: 1e12.
    
    Returns:
        knn_indices: torch.LongTensor (N, max_neighbor), the indices in `s_points` of the neighbors of each query point.
            If there are less than `max_neighbor` neighbors, the indices are filled with M.
    """
    batch_size = len(q_lengths)
    q_start_index = 0
    s_start_index = 0
    knn_func = keops_knn(max_neighbor)
    knn_indices_list = []
    for i in range(batch_size):
        cur_q_length = q_lengths[i]
        cur_s_length = s_lengths[i]
        q_end_index = q_start_index + cur_q_length
        s_end_index = s_start_index + cur_s_length
        cur_q_points = q_points[q_start_index:q_end_index]
        cur_s_points = s_points[s_start_index:s_end_index]
        knn_distances, knn_indices = knn_func(cur_q_points, cur_s_points)
        knn_indices = knn_indices + s_start_index
        knn_masks = torch.lt(knn_distances, radius ** 2)
        knn_indices.masked_fill_(~knn_masks, s_points.shape[0])
        knn_indices_list.append(knn_indices)
        q_start_index = q_end_index
        s_start_index = s_end_index
    knn_indices = torch.cat(knn_indices_list, dim=0)
    return knn_indices


@torch.no_grad()
def grid_subsample(points, lengths, voxel_size):
    r"""Batch grid subsampling.

    Subsample the points with `voxel_size`.

    Args:
        points: torch.Tensor (N, 3), the point clouds in the batch.
        lengths: list of ints (B,), the numbers of points in the batch.
        voxel_size: float, voxel size.
    
    Returns:
        sampled_points: torch.Tensor (M, 3), the sampled point clouds in the batch.
        sampled_lengths: list of ints (B,), the numbers of sampled points in the batch.
    """
    batch_size = len(lengths)
    start_index = 0
    sampled_points_list = []
    for i in range(batch_size):
        length = lengths[i]
        end_index = start_index + length
        points = points[start_index:end_index]
        min_corner = points.amin(0)
        max_corner = points.amax(0)
        origin = torch.floor(min_corner / voxel_size) * voxel_size
        x_size = int(np.floor((max_corner[0].item() - origin[0].item()) / voxel_size)) + 1
        y_size = int(np.floor((max_corner[1].item() - origin[1].item()) / voxel_size)) + 1
        coords = torch.floor((points - origin) / voxel_size).long()
        coord_indices = coords[:, 2] * (x_size * y_size) + coords[:, 1] * x_size + coords[:, 0]
        _, inv_indices, unique_counts = torch.unique(coord_indices, return_inverse=True, return_counts=True)
        inv_indices = inv_indices.unsqueeze(1).expand(-1, 3)
        sampled_points = torch.zeros(unique_counts.shape[0], 3).to(points.device)
        sampled_points.scatter_add_(0, inv_indices, points)
        sampled_points /= unique_counts.unsqueeze(1).float()
        sampled_points_list.append(sampled_points)
        start_index = end_index
    sampled_points = torch.cat(sampled_points_list, dim=0)
    sampled_lengths = [x.shape[0] for x in sampled_points_list]
    return sampled_points, sampled_lengths


@torch.no_grad()
def create_grid_pyramid(points, lengths, num_level, init_voxel_size, init_search_radius, max_neighbors):
    r"""Grid subsampling and neighbor searching.

    Progresstively subsample a batch of point clouds with grid subsampling and search the neighbors for convolution.

    The input point clouds are already subsampled with `init_voxel_size`. The voxel size is doubled in each downsampling
    operation. The search radius for neighbors at the first level (input point clouds) is `init_search_radius`, and it
    doubled after each downsampling operation. 

    Args:
        points: torch.Tensor (N, 3), the input point clouds in the batch.
        lengths: list of int (B,), the numbers of input points in the batch.
        num_level: int, the number of downsampling level.
        init_voxel_size: float.
        init_search_radius: float.
        max_neighbors: list of int (num_level,), the maximum number of neighbors in each level.
    
    Returns:
        pyramid_dict: a `dict` contains the point cloud pyramid.
            `points`: list of torch.Tensor (N_i, 3), the point clouds in each level.
            `lengths`: list of list of int (B,), the number of points in each level.
            `neighbors`: list of torch.LongTensor (N_i, max_neighbors[i]), the neighbors in each level.
            `subsampling`: list of torch.LongTensor (N_i, max_neighbors[i]), the subsampling indices form $i$-th level
                to $(i+1)$-th level.
            `subsampling`: list of torch.LongTensor (N_i, max_neighbors[i]), the upsampling indices from $i+1$-th level
                to $i$-th level.

    """
    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    voxel_size = init_voxel_size
    for i in range(num_level):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    search_radius = init_search_radius
    for i in range(num_level):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = batch_radius_search(
            cur_points, cur_points, cur_lengths, cur_lengths, search_radius, max_neighbors[i]
        )
        neighbors_list.append(neighbors)

        if i < num_level - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = batch_radius_search(
                sub_points, cur_points, sub_lengths, cur_lengths, search_radius, max_neighbors[i]
            )
            subsampling_list.append(subsampling)

            upsampling = batch_radius_search(
                cur_points, sub_points, cur_lengths, sub_lengths, search_radius * 2, max_neighbors[i + 1]
            )
            upsampling_list.append(upsampling)

        search_radius *= 2

    pyramid_dict = {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list
    }

    return pyramid_dict
