import time

import torch
import torch.nn as nn

from .functional import grid_subsample, radius_search, create_grid_pyramid


class RadiusSearch(nn.Module):
    def __init__(self, radius, max_neighbor):
        super(RadiusSearch, self).__init__()
        self.radius = radius
        self.max_neighbor = max_neighbor

    def forward(self, q_points, s_points, q_lengths, s_lengths):
        return radius_search(q_points, s_points, q_lengths, s_lengths, self.radius, self.k)

    def __repr__(self):
        return self.__class__.__name__ + '(radius={:.3f}, k={})'.format(self.radius, self.k)


class GridSubsample(nn.Module):
    def __init__(self, voxel_size):
        super(GridSubsample, self).__init__()
        self.voxel_size = voxel_size

    def forward(self, stacked_points, stacked_lengths):
        return grid_subsample(stacked_points, stacked_lengths, self.voxel_size)

    def __repr__(self):
        return self.__class__.__name__ + '(voxel_size={:.3f})'.format(self.voxel_size)


class GridPyramid(nn.Module):
    def __init__(self, num_level, voxel_size, search_radius, max_neighbors):
        super(GridPyramid, self).__init__()
        self.num_level = num_level
        self.voxel_size = voxel_size
        self.search_radius = search_radius
        self.max_neighbors = max_neighbors
        if self.num_level != len(self.max_neighbors):
            raise ValueError('The value of "num_level" and the size of "max_neighbors" do not match.')

    def forward(self, points, lengths):
        pyramid = create_grid_pyramid(
            points, lengths, self.num_level, self.voxel_size, self.search_radius, self.max_neighbors
        )
        return pyramid
